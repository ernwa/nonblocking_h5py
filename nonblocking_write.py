from __future__ import print_function
import multiprocessing
import numpy as np
import h5py
import ctypes
import sys, time, os

N_BUFS = 16
N_CHUNKS = 1
N_FRAMES = 1000

FRAME_SHAPE = (1024,1024)
BUF_SHAPE = (N_BUFS,) + FRAME_SHAPE
CHUNK_SHAPE = (N_CHUNKS,) + FRAME_SHAPE
DATASET_SHAPE = (N_FRAMES,) + FRAME_SHAPE


def get_buffer_address(mv):
    return ctypes.addressof(ctypes.c_char.from_buffer(mv))


class h5pyNonblocking(object):
    @staticmethod
    def save_worker( h5_path, ds_name, pipe, shared_memory):
        with h5py.File(h5_path, 'a', libver='latest') as f:
            dataset = f[ds_name]
            for dest_slice, (shape, dtype, offset) in iter(pipe.recv, None):
                save_array = np.ndarray( shape, dtype, buffer=shared_memory, offset=offset )
    #            print('writing slice ', dest_slice, save_array.shape)
                dataset.write_direct( save_array, dest_sel=dest_slice )       # blocks here till write done
    #            print('done', offset + save_array.nbytes)
                pipe.send((offset, save_array.nbytes))

        pipe.send(None)


    def __init__(self, h5_path, ds_name, buffer_size=(128 * 1024*1024), external_buf=None, callback=None, on_overflow="fail" ):
        self.h5_path = h5_path
        self.ds_name = ds_name
        self.zero_copy_mode = (external_buf is not None)
        self.write_callback = callback
        self.on_overflow = on_overflow

        if self.zero_copy_mode:
            # use posix strategy of read
            if isinstance(external_buf, np.ndarray):
                self.shared_buf = memoryview(external_buf)
                self.sz_buffer = len(self.shared_buf.cast('B'))
#                print(self.sz_buffer)
            else:
                self.shared_buf = external_buf
                self.sz_buffer = len(external_buf)

            self.ptr_buffer = get_buffer_address( external_buf )
        else:
            self.sz_buffer = int(buffer_size)
            self.shared_buf = multiprocessing.RawArray('B', self.sz_buffer)
            self.i_head, self.i_tail = 0, -1


        self.n_writes_waiting = 0

        self.pipe, self.worker_pipe = multiprocessing.Pipe()
        worker_args = self.h5_path, self.ds_name, self.worker_pipe, self.shared_buf
        self.writer = multiprocessing.Process( target=self.save_worker, args=worker_args )
        self.writer.start()


    def __enter__(self):
        return self

    def __exit__(self, extype, exvalue, extraceback):
        self.close()
        return extype is None


    def close(self):
        if self.writer.is_alive():
            self.pipe.send(None)
            for obj in iter(self.pipe.recv, None):
                pass
            self.writer.join()


    def get_completed_write(self):
        i_obj, sz_obj = self.pipe.recv()
        if self.write_callback:
            self.write_callback(i_obj, sz_obj)
        self.n_writes_waiting -= 1
        return i_obj, sz_obj


    def __setitem__(self, slicelist, obj):
        i_obj, sz_obj = 0, 0
        while self.pipe.poll():                 # don't block, just take what's there
            i_obj, sz_obj = self.get_completed_write()

        if self.zero_copy_mode:
            np_base_addr = obj.ctypes.data
            if not (self.ptr_buffer <= np_base_addr < (self.ptr_buffer + self.sz_buffer)):
                raise ValueError("Zero-copy mode can only queue writes from within external buffer!")
            buf_offset = np_base_addr - self.ptr_buffer
            buf_obj = obj
        else:
            while True:
                overflow = False
                if i_obj:
                    self.i_tail = i_obj + sz_obj

                if self.i_head + obj.nbytes > self.sz_buffer:
                    if self.i_head < self.i_tail:
                        overflow = True
                    buf_offset = 0
                else:
                    buf_offset = self.i_head

                new_head = buf_offset + obj.nbytes
    #            print('queuing [%08x:%08x]. tail=%08x' % (self.i_head, new_head, self.i_tail))
                if buf_offset <= self.i_tail < new_head:   #TODO: is this best way?
                    overflow = True

                if not overflow:
                    break
                else:
                    if self.on_overflow == "fail":
                        raise RuntimeError("write buffer is full!")
                    elif self.on_overflow == "wait":
                        i_obj, sz_obj = self.get_completed_write()          # block until space clears up
                    else:
                        raise ValueError("on_overflow must be either 'fail' or 'wait' ")

            buf_obj = np.ndarray( obj.shape, obj.dtype,
                        buffer=self.shared_buf, offset=buf_offset )
            np.copyto( buf_obj, obj )
            self.i_head = new_head

        self.n_writes_waiting += 1
        self.pipe.send( (slicelist, (buf_obj.shape, buf_obj.dtype, buf_offset) ) )



import tempfile

def create_file_and_dset(shape, dn='test', *args, **kwargs):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        fn = f.name
    with h5py.File(fn, 'w', libver='latest') as f:
        f.create_dataset(dn, shape, *args, **kwargs)
    return fn, dn


def verify_file(fn, check_arr, dn='test' ):
    check_w = check_arr.shape[0]
    with h5py.File(fn, 'r') as f:
        ds = f[dn]
        n_loops = int( np.ceil( ds.shape[0] / float(check_arr.shape[0]) ) )
        for i in range(0, n_loops):
            i_s = i * check_w
            i_e = i_s + check_w
            chunk = ds[i_s:i_e]
            w = chunk.shape[0]
            print('%d: verifying file[%d:%d] == check_arr[0:%d]' % (i, i_s, i_s + w, w))
            check = chunk == check_arr[:w]

            if not np.all(check):
                for i_c in range(w):
                    n_errs = np.sum(check[i_c], dtype='u4')
                    if n_errs:
                        print( '%d incorrect values in frame %d ' % (n_errs, i_s + i_c))

#            assert np.all( chunk == check_arr[:w] )
    os.unlink(fn)


def create_data(shape):
    return np.random.randint(0, 65535, shape, dtype='u2')


def free_func(ptr, sz):
    print("saved %d byte buffer segment [%08x:%08x] " % (sz, ptr, ptr+sz))


def test_external_buffer_write():
    buf = create_data( BUF_SHAPE )
    fn, dsn = create_file_and_dset( DATASET_SHAPE, chunks=CHUNK_SHAPE, dtype=buf.dtype, shuffle=True )


    with h5pyNonblocking(fn, dsn, external_buf=buf, callback=free_func, on_overflow='wait') as writer:
        for i in range(N_FRAMES):
            writer[i] = buf[i % N_BUFS]
#            time.sleep(0.01)

    verify_file( fn, buf )


def test_internal_buffer_write():
    buf = create_data( BUF_SHAPE )
    fn, dsn = create_file_and_dset( DATASET_SHAPE, chunks=CHUNK_SHAPE, dtype=buf.dtype, shuffle=True )

    with h5pyNonblocking(fn, dsn, callback=free_func, on_overflow='wait') as writer:
        for i in range(N_FRAMES):
            writer[i] = buf[i % N_BUFS]
#            time.sleep(0.01)

    verify_file( fn, buf )


if __name__ == "__main__":
    # test_external_buffer_write()
    # test_internal_buffer_write()
    # sys.exit()
    import matplotlib.pyplot as plt
    import logging
    logger = logging.getLogger('matplotlib')
    logger.setLevel(logging.WARNING)
#    N_FRAMES = 1000
    fn = 'test2.hdf5'

    print( FRAME_SHAPE, CHUNK_SHAPE)
    with h5py.File(fn, 'w', libver='latest') as f:
        try:
            ds = f.create_dataset('ztest', DATASET_SHAPE, dtype='u4', chunks=CHUNK_SHAPE, shuffle=True, compression='lzf' )
        except RuntimeError:
            ds = f['ztest']


    buf = create_data(BUF_SHAPE)

    with h5pyNonblocking(fn, 'ztest', external_buf=buf) as dataset_writer:
#        if True:
#            dataset_writer = ds
        times = np.zeros((N_FRAMES, 2), 'f8')
        depth = np.zeros(N_FRAMES, 'u2')

        for i in range(N_FRAMES):
            times[i,0] = time.time()
            dataset_writer[i] = buf[i % N_BUFS]
            times[i,1] = time.time()
            depth[i] = dataset_writer.n_writes_waiting
            if dataset_writer.zero_copy_mode:
                print('write %3d: %2d items in queue' % (i, depth[i]))
            else:
                print('write %3d: %2d items in queue; head=%08x, tail=%08x' % (i, depth[i], dataset_writer.i_head, dataset_writer.i_tail ))

            time.sleep(0.020)
        print('waiting on thread to flush')

    plt.figure()
    plt.title('nonblocking hdf5 write test')

    plt.subplot(211)
    plt.plot(np.diff(times, axis=0)[:,0] * 1000, label='write loop time')
    plt.plot(np.diff(times, axis=1) * 1000, label='write call time')
    plt.legend()
    plt.xlabel('write #')
    plt.ylabel('msec')

    plt.subplot(212)
    plt.plot(depth, label='waiting')
    plt.legend()
    plt.xlabel('write #')
    plt.ylabel('queue size (writes)')
    plt.show()
