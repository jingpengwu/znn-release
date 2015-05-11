from libcpp.list cimport list
from libcpp.string cimport string
from libc.stdint import uintptr_t

cdef extern from "network.hpp" namespace "zi::znn":
    cdef cppclass network:
        network( options_ptr ) except +
        prepare_training()
        set_minibatch_size( vec3i )
        set_minibatch_size( double )
        setup_fft()
        train()
        forward_scan()
        test_check()
        prepare_testing()
        export_train_information()
        run_forward( list[<uintptr_t>double3d_ptr] )
        run_backward( list[uintptr_t], list[uintptr_t(double3d_ptr)] )


cdef extern from "options.hpp" namespace "zi::znn, std":
    cdef cppclass options:
        options( const string& ) except +
        # [PATH]
        string      config_path;
        string      load_path;
        string      data_path;
        string      save_path;
        string      hist_path;

    	# [OPTIMIZE]
        std::size_t        n_threads;
    	bool               force_fft;
    	bool               optimize_fft;

    	# [TRAIN]
    	batch_list         train_range;
    	batch_list         test_range;
    	vec3i              outsz;
    	string             dp_type;
    	string             cost_fn;
    	double             cost_fn_param;
    	bool               data_aug;
    	double             cls_thresh;
    	bool               softmax;

    	# [UPDATE]
    	double             force_eta;
    	double             momentum;
    	double             wc_factor;
    	double             anneal_factor;
    	std::size_t        anneal_freq;
    	bool               minibatch;
    	bool               norm_grad;
    	bool               rebalance;

    	# [MONITOR]
    	std::size_t        n_iters;
    	std::size_t        check_freq;
    	std::size_t        test_freq;
    	std::size_t        test_samples;

    	# [SCANNING]
    	string             scanner;
    	vec3i              scan_offset;
    	vec3i              subvol_dim;
    	std::size_t        weight_idx;
    	bool               force_load;
    	bool               out_filter;
    	string             outname;
    	string             subname;

        # functions
        save()
        bool build( const string& )
        batch_list get_batch_range()
        cost_fn_ptr create_cost_function()
        path_check()
