# LUMI training - Omnitrace by Example - Oslo, Norway - June 2024


## MPI Ghost Exchange Optimization Examples
This set of examples is meant to provide a route towards the incremental optimization of a ghost exchange cells. 

### Changes Between Example Versions
This code contains several implementations of the same ghost exchange algorithm at varying stages
of optimization:
- **Orig**: Shows a CPU-only implementation that uses MPI, and serves as the starting point for further optimizations. It is recommended to start here!
- **Ver1**: Shows an OpenMP target offload implementation that uses the Managed memory model to port the code to GPUs using host allocated memory for MPI communication.
- **Ver2**: Shows the usage and advantages of using `roctx` ranges to get more easily readable profiling output from Omnitrace.
- **Ver3**: Under Construction, not expected to work at the moment
- **Ver4**: Explores heap-allocating communication buffers once on host.
- **Ver5**: Explores unrolling a 2D array to a 1D array.
- **Ver6**: Explores using explicit memory management directives to specify when data movement should happen.
- **Ver7**: Under Construction, not expected to work at this time.

<details> <summary><h3>Background Terminology: We're Exchanging <i>Ghosts?</i></h3></summary>
<h4>Problem Decomposition</h4>
In a context where the problem we're trying to solve is spread across many compute resources,
it is usually inefficient to store the entire data set on every compute node working to solve our problem.
Thus, we "chop up" the problem into small pieces we assign to each node working on our problem.
Typically, this is referred to as a <b>problem decomposition</b>.<br/>
<h4>Ghosts, and Their Halos</h4>
In problem decompositions, we may still need compute nodes to be aware of the work that other nodes
are currently doing, so we add an extra layer of data, referred to as a <b>halo</b> of <b>ghosts</b>.
This region of extra data can also be referred to as a <b>domain boundary</b>, as it is the <b>boundary</b>
of the compute node's owned <b>domain</b> of data.
We call it a <b>halo</b> because typically we need to know all the updates happening in the region surrounding a single compute node's data.
These values are called <b>ghosts</b> because they aren't really there: ghosts represent data another
 compute node controls, and the ghost values are usually set unilaterally through communication
between compute nodes.
This ensures each compute node has up-to-date values from the node that owns the underlying data.
These updates can also be called <b>ghost exchanges</b>.
</details>

### Overview of the Ghost Exchange Implementation
The implementations presented in these examples follow the same basic algorithm.
They each implement the same computation, and set up the same ghost exchange, we just change where computation happens, or specifics with data movement or location.

The code is controlled with the following arguments:
- `-i imax -j jmax`: set the total problem size to `imax*jmax` elements.
- `-x nprocx -y nprocy`: set the number of MPI ranks in the x and y direction, with `nprocx*nprocy` total processes.
- `-h nhalo`: number of halo layers, typically assumed to be 1 for our diagrams.
- `-t (0|1)`: whether time synchronization should be performed.
- `-c (0|1)`: whether corners of the ghost halos should also be communicated.

The computation done on each data element after setup is a blur kernel, that modifies the value of a given element by averaging the values at a 5-point stencil location centered at the given element:

`xnew[j][i] = (x[j][i] + x[j][i-1] + x[j][i+1] + x[j-1][i] + x[j+1][i])/5.0`

The communication pattern used is best shown in a diagram that appears in [Parallel and high performance computing, by Robey and Zamora](https://www.manning.com/books/parallel-and-high-performance-computing):

![ghost_exchange2](https://hackmd.io/_uploads/SJWWc4o4A.png)

In this diagram, a ghost on a process is represented with a dashed outline, while owned data on a process is represented with a solid line. Communication is represented with arrows and colors representing the original data, and the location that data is being communicated and copied to. We see that each process communicates based on the part of the problem it owns: the process that owns the central portion of data must communicate in all four directions, while processes on the corner only have to communicate in two directions.

## Ghost Exchange: Original Implementation

This example shows a CPU-only implementation, and how to use Omnitrace to trace it.

### Environment for LUMI

```
module load CrayEnv
module load buildtools/23.09

module load PrgEnv-cray/8.4.0
module load cce/16.0.1
module load craype-accel-amd-gfx90a
module load craype-x86-trento

module use /pfs/lustrep2/projappl/project_462000125/samantao-public/mymodules

module load rocm/5.4.3 omnitrace/1.11.2-rocm-5.4.x
```
You can setup the following environment variables for the project you want to use:
```
export SALLOC_ACCOUNT=project_<your porject ID>
export SBATCH_ACCOUNT=project_<your porject ID>
```
### Download Material

```
git clone https://github.com/amd/HPCTrainingExamples.git
cd HPCTrainingExamples/MPI-examples/GhostExchange/GhostExchange_ArrayAssign
```

### Building and Running

To build and run this initial implementation do the following:

```
cd Orig
mkdir build; cd build;
cmake ..
nice make -j
srun -N1 -n4 -c7 --gpu-bind=closest -t 05:00 ./GhostExchange -x 2 -y 2 -i 20000 -j 20000 -h 2 -t -c -I 100
```

Output from this run should look something like this:

```
GhostExchange_ArrayAssign Timing is stencil 12.035283 boundary condition 0.027232 ghost cell 0.051823 total 12.440170
```

### Instrumenting with Binary-rewrite

Before instrumenting and running with Omnitrace, we need to make sure our default configuration file is generated with:

```
srun -n1 omnitrace-avail -G ~/.omnitrace.cfg
export OMNITRACE_CONFIG_FILE=~/.omnitrace.cfg
```

Note that `~/.omnitrace.cfg` is the default place Omnitrace will look for a configuration file, but
you can point it to a different configuration file using the environment variable `OMNITRACE_CONFIG_FILE`.

It is recommended to use `omnitrace-instrument` to output an instrumented binary since our application uses MPI. This way, tracing output will appear in separate files by default. We can instrument and run with these commands:

```
omnitrace-instrument -o ./GhostExchange.inst -i 100 -- ./GhostExchange
srun -N1 -n4 -c7 --gpu-bind=closest -t 05:00 omnitrace-run -- ./GhostExchange.inst -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 100
```

Note: it is necessary to run with `omnitrace-run` when running an instrumented binary.

### Expected Instrumentation Output

Omnitrace will output text indicating its progress when using both `omnitrace-instrument` and
`omnitrace-run`. `omnitrace-instrument` shows which functions it instrumented and which functions are available to be instrumented in output files, the paths are reported as shown here:

![instrument_output](https://hackmd.io/_uploads/B1HBiEoER.png)

The `available` output file looks like this:
```
  StartAddress   AddressRange  #Instructions  Ratio Linkage Visibility  Module                                                                                   Function                         FunctionSignature               
      0x204a8c              9              3   3.00    local     hidden ../sysdeps/x86_64/crti.S                                                                 _fini                            _fini                           
      0x204a74             23              7   3.29    local     hidden ../sysdeps/x86_64/crti.S                                                                 _init                            _init                           
      0x2026be             45             13   3.46   global    default ../sysdeps/x86_64/start.S                                                                _start                           _start                          
      0x204450           1166            308   3.79  unknown    unknown /users/samantao/amd/scratch/lumi-training-oslo2024/HPCTrainingExamples/MPI-ex...         Cartesian_print                  Cartesian_print                 
      0x2038b0            740            203   3.65  unknown    unknown /users/samantao/amd/scratch/lumi-training-oslo2024/HPCTrainingExamples/MPI-ex...         boundarycondition_update         boundarycondition_update        
      0x203ba0           2222            551   4.03  unknown    unknown /users/samantao/amd/scratch/lumi-training-oslo2024/HPCTrainingExamples/MPI-ex...         ghostcell_update                 ghostcell_update                
      0x2035f0            694            175   3.97  unknown    unknown /users/samantao/amd/scratch/lumi-training-oslo2024/HPCTrainingExamples/MPI-ex...         haloupdate_test                  haloupdate_test                 
      0x202780           3103            698   4.45   global    default /users/samantao/amd/scratch/lumi-training-oslo2024/HPCTrainingExamples/MPI-ex...         main                             main                            
      0x2033a0            591            157   3.76  unknown    unknown /users/samantao/amd/scratch/lumi-training-oslo2024/HPCTrainingExamples/MPI-ex...         parse_input_args                 parse_input_args                
      0x204940            129             43   3.00   global    default /users/samantao/amd/scratch/lumi-training-oslo2024/HPCTrainingExamples/MPI-ex...         malloc2D                         malloc2D                        
      0x2049d0             15              4   3.75   global    default /users/samantao/amd/scratch/lumi-training-oslo2024/HPCTrainingExamples/MPI-ex...         malloc2D_free                    malloc2D_free                   
      0x2048e0             13              3   4.33   global    default /users/samantao/amd/scratch/lumi-training-oslo2024/HPCTrainingExamples/MPI-ex...         cpu_timer_start                  cpu_timer_start                 
      0x2048f0             71             19   3.74   global    default /users/samantao/amd/scratch/lumi-training-oslo2024/HPCTrainingExamples/MPI-ex...         cpu_timer_stop                   cpu_timer_stop                  
      0x202729             82             22   3.73    local    default GhostExchange                                                                            __do_fini                        __do_fini                       
      0x2026eb             62             17   3.65    local    default GhostExchange                                                                            __do_init                        __do_init                       
      0x204a6e              4              3   1.33   global    default elf-init.c                                                                               __libc_csu_fini                  __libc_csu_fini                 
      0x2049ee            103             36   2.86   global    default elf-init.c                                                                               __libc_csu_init                  __libc_csu_init   
```
<!--![available](https://hackmd.io/_uploads/BJZ8jNj4A.png)-->

While the `instrumented` output file looks like this:
```
  StartAddress   AddressRange  #Instructions  Ratio Linkage Visibility  Module                                                                                   Function                         FunctionSignature               
      0x204450           1166            308   3.79  unknown    unknown /users/samantao/amd/scratch/lumi-training-oslo2024/HPCTrainingExamples/MPI-ex...         Cartesian_print                  Cartesian_print                 
      0x2038b0            740            203   3.65  unknown    unknown /users/samantao/amd/scratch/lumi-training-oslo2024/HPCTrainingExamples/MPI-ex...         boundarycondition_update         boundarycondition_update        
      0x203ba0           2222            551   4.03  unknown    unknown /users/samantao/amd/scratch/lumi-training-oslo2024/HPCTrainingExamples/MPI-ex...         ghostcell_update                 ghostcell_update                
      0x2035f0            694            175   3.97  unknown    unknown /users/samantao/amd/scratch/lumi-training-oslo2024/HPCTrainingExamples/MPI-ex...         haloupdate_test                  haloupdate_test                 
      0x2033a0            591            157   3.76  unknown    unknown /users/samantao/amd/scratch/lumi-training-oslo2024/HPCTrainingExamples/MPI-ex...         parse_input_args                 parse_input_args 
```
<!--![instrumented](https://hackmd.io/_uploads/rJ68jVjVC.png)-->


We see in this case `omnitrace-instrument` seems to only instrument a few functions. This is because by default Omnitrace excludes any functions smaller than a certain number of instructions from instrumentation to reduce the overhead of tracing, the size of the resulting trace, and increase readability of the trace visualization. This can be tuned with the `-i <instruction-count>` argument to `omnitrace-instrument`, which will include functions with at least `<instruction-count>` instructions in instrumentation. We used the `-i 100` option to instrument functions with more than 100 instructions. Specific functions can be included by providing a regular expression to the `-I <function-regex>`, which will include in instrumentation any function name matching the regular expression, despite heuristics.

For more thorough details on Omnitrace options, we defer to the [Omnitrace documentation](https://rocm.github.io/omnitrace).

For `omnitrace-run`, we look for the following output to ensure our run is correctly using Omnitrace, and for locating the output files:
![ascii_omni](https://hackmd.io/_uploads/S1JdsVsVR.png)


The ASCII art lets you know Omnitrace is running, and:
![output_paths](https://hackmd.io/_uploads/By_OjVoER.png)


Shows the output paths for the proto files, and also validates that the proto files generated successfully.

If the `omnitrace-run` output seems to halt abruptly without the output file paths, ensure your app can run successfully outside of Omnitrace.

### Initial Trace

Below is a screenshot of a trace obtained for this example:
![orig_0](https://hackmd.io/_uploads/rJ15s4sEA.png)

(truncated for space)
![orig_1](https://hackmd.io/_uploads/rJ85jNj4R.png)

In this screenshot, we see Omnitrace is showing CPU frequency data for every core.
To have Omnitrace only show CPU frequency for a single CPU core, add this to `~/.omnitrace.cfg`:

```
OMNITRACE_SAMPLING_CPUS                            = 0
```

and re-run the command from before, no need to re-instrument:

```
srun -N1 -n4 -c7 --gpu-bind=closest -t 05:00 omnitrace-run -- ./GhostExchange.inst -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 100
```

Now we see that only one instance of CPU frequency is reported:

![orig_3_sample_1cpu](https://hackmd.io/_uploads/rJPos4sVR.png)

Zooming in, we see instrumented MPI activity:
![orig_2_zoom_in](https://hackmd.io/_uploads/rJiasVoNR.png)



We can alter the Omnitrace configuration to see MPI overheads measured numerically. Add this to `~/.omnitrace.cfg`:

```
OMNITRACE_PROFILE                                  = true
```

Then re-running our same instrumented binary gives us a few new output files, look for `wall_clock-0.txt`:
![profile](https://hackmd.io/_uploads/r1TAjVsNC.png)


Here, we see a hierarchical view of overheads, to flatten the profile to see total count and mean duration for each MPI call, add this to `~/.omnitrace.cfg`:

```
OMNITRACE_FLAT_PROFILE                             = true
```

Re-running `omnitrace-run` with our intrumented binary will now produce a `wall_clock-0.txt` file that looks like this:
![flat_profile](https://hackmd.io/_uploads/SJL13VoN0.png)


We can see the number of times each MPI function was called, and the time associated with each.

## Ghost Exchange Version 1: OpenMP GPU port

In this version of the Ghost Exchange, we port the initial code to GPUs by using OpenMP pragmas. This uses OpenMP's target offload feature with a managed memory model, so the only differences between the original code and this version are the addition of OpenMP pragmas.

Using the managed memory model, the memory buffers are still initially allocated on host, but the OS will manage page migration and data movement across the Infinity Fabric&trade; link on MI250X.


### Build and Run

```
cd Ver1
mkdir build; cd build;
cmake ..
make -j8
export HSA_XNACK=1
srun -N1 -n4 -c7 --gpu-bind=closest -t 05:00 ./GhostExchange -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 100
```


This run should show output that looks like this:

```
GhostExchange_ArrayAssign Timing is stencil 18.453013 boundary condition 0.019862 ghost cell 0.031657 total 19.063382
```

Now, this runtime is somewhat unexpected and points to some issue in our OpenMP configuration. Certainly, using managed memory means sub-optimal memory movement, but we have observed on a different system that this implementation runs in around 3 seconds. Using different compilers makes matching configurations complex, so we suspect there is some subtle configuration difference that is impacting performance here.

### Initial Trace

Remember to enable the `HSA_XNACK` environment variable and ensure that the configuration file is known to Omnitrace:

```
export HSA_XNACK=1
export OMNITRACE_CONFIG_FILE=~/.omnitrace.cfg

omnitrace-instrument -o ./GhostExchange.inst -I boundarycondition_update -- ./GhostExchange

srun -N1 -n4 -c7 --gpu-bind=closest -t 05:00 omnitrace-run -- ./GhostExchange.inst -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 100
```
### Pin Rows for Easier Visualization

The profile we obtain will show all the offloaded kernels launched with the OpenMP pragmas.
However, the initial visualization we get with Perfetto often does not group the rows of interest
together. In this screenshot, we show the "pin" icon that allows us to pin rows to the top of the
visualization, allowing us to see relevant tracing data for our specific case:

![pinned_visualization](https://hackmd.io/_uploads/SkH0hEj4R.png)

This profile will also show `rocm-smi` information about each GPU, though that data seems to indicate only GPU 0 is engaged. To show only information relevant to GPU 0, we can add this to `~/.omnitrace.cfg`:

```
OMNITRACE_SAMPLING_GPUS                            = 0
```

Before this is set, the profile looks like this:
![too_many_gpus](https://hackmd.io/_uploads/H1IkaEi4R.png)


And after we re-run `omnitrace-run` with `OMNITRACE_SAMPLING_GPUS=0`, we see:
![only_one_gpu](https://hackmd.io/_uploads/BJDgT4jNR.png)


### Look at the Flat Timemory profile

Again, add `OMNITRACE_PROFILE=true` and `OMNITRACE_FLAT_PROFILE=true` to `~/.omnitrace.cfg` to get `wall_clock-0.txt` to see overall overhead in seconds for each function:

![timemory_flat](https://hackmd.io/_uploads/SJQZTNs4R.png)


We now see kernels with `__omp_offloading...` that show we are launching kernels, and we see that our runtime for `GhostExchange.inst` has increased to about 20 seconds. We also see that the only function call that takes around that long in the profile is `hipStreamSynchronize`. This indicates that the bulk of the time is spent in the GPU compute kernels, especially the blur kernel. We know that the kernel is memory bound, but the very long duration indicates that there is an overhead due to page migration. Omnitrace has helped us narrow down where this overhead is coming from, but it does not show page faults arising from GPU kernels. We are hoping that this feature would be available in a future update.

### HSA Activity


The AMD compiler implements OpenMP target offload capability using the [HSA](https://rocm.docs.amd.com/projects/ROCR-Runtime/en/latest/index.html) runtime library. The Cray compiler, on the other hand, implements OpenMP target offload functionality using the [HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/index.html) runtime. Ultimately, the HIP runtime relies on the HSA runtime. To see HSA activity, add this to `~/.omnitrace.cfg`:

```
OMNITRACE_ROCTRACER_HSA_ACTIVITY                   = true
OMNITRACE_ROCTRACER_HSA_API                        = true
```

This will add HSA layer activities in the trace:
![hsa_trace](https://hackmd.io/_uploads/S1Of6NjNC.png)

## Ghost Exchange Version 2: `roctx` ranges

In this implementation, we add `roctx` ranges to the code to demonstrate their use in highlighting regions of interest in a trace. This allows you to group several functions and kernels in the same range to mark logical regions of your application, making the trace easier to understand at a higher level.

### Build and Run
```
cd Ver2
mkdir build; cd build;
cmake ..
nice make -j
srun -N1 -n4 -c7 --gpu-bind=closest -t 05:00 ./GhostExchange -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 100
```

Output of this run should look like this:

```
GhostExchange_ArrayAssign Timing is stencil 18.446859 boundary condition 0.019800 ghost cell 0.025591 total 19.056449
```

Again, we see an unexpectedly high runtime for a GPU port, likely due to some OpenMP configuration detail. On a different system, this runs in around 3 seconds.

#### Get an Initial Trace

Omnitrace enables `roctx` ranges by enabling in our configuration file the following:
```
OMNITRACE_USE_ROCTX = true
```
With that, we only needed to add the relevant instrumentation in the code.
Instrument and run the instrumented application to get a trace:

```
export HSA_XNACK=1
export OMNITRACE_CONFIG_FILE=~/.omnitrace.cfg
omnitrace-instrument -o ./GhostExchange.inst -- ./GhostExchange
srun -N1 -n4 -c7 --gpu-bind=closest -t 05:00 omnitrace-run -- ./GhostExchange.inst -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 100
```

With `roctx` regions, our trace looks like this:
![roctx_trace](https://hackmd.io/_uploads/SJWHYIiVC.png)


Note: to reduce the number of rows of `rocm-smi` output you see, you may also need to add `OMNITRACE_SAMPLING_GPUS=0`

### Look at Timemory output

With `OMNITRACE_PROFILE=true` and `OMNITRACE_FLAT_PROFILE=true` in your `~/.omnitrace.cfg` you will see a `wall_clock-0.txt` file that looks like this:

![timemory_output_ver2](https://hackmd.io/_uploads/B1T-1_jEA.jpg)

We see that `roctx` regions also show up in this file. Importantly we see the region called `BufAlloc` gets called 101 times, showing that the code allocating our buffers is running multiple times throughout our application's execution.
                                             

## Ghost Exchange Version 4: Reduce Allocations

In the first `roctx` range example we saw that BufAlloc was being called 101 times, indicating we were allocating our buffers several times. In this example, we move the allocations so that we only need to allocate the buffers one time and explore how that impacts performance through Omnitrace.


### Build and Run

```
cd Ver4
mkdir build; cd build;
cmake ..
nice make -j
srun -N1 -n4 -c7 --gpu-bind=closest -t 05:00 ./GhostExchange -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 100
```

The output for this run should look like:

```
GhostExchange_ArrayAssign Timing is stencil 18.451170 boundary condition 0.019720 ghost cell 0.046104 total 19.078582
```

Note we see similar runtimes to previous examples, so these changes do not fix the issue.

### Get an Initial Trace

```
export HSA_XNACK=1
export OMNITRACE_CONFIG_FILE=~/.omnitrace.cfg
omnitrace-instrument -o ./GhostExchange.inst -- ./GhostExchange
srun -N1 -n4 -c7 --gpu-bind=closest -t 05:00 omnitrace-run -- ./GhostExchange.inst -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 100
```

This trace should look largely like the previous roctx trace:
![roctx_trace](https://hackmd.io/_uploads/HJ7lqUjV0.png)


An easier way to see how this code has changed is to look at `wall_clock-0.txt`, by adding
`OMNITRACE_PROFILE=true` and `OMNITRACE_FLAT_PROFILE=true` to `~/.omnitrace.cfg`:



Here we see that the change has the intended effect of reducing the number of calls to `BufAlloc` to one, rather than 101.

## Ghost Exchange Version 5: Changing Data Layout

In this example we explore changing our 2D array layout to 1D and use Omnitrace to investigate the performance impact.

This sort of change typically requires significant development overhead, as the indexing of the data must change everywhere in the application.

### Build and Run

```
cd Ver5
mkdir build; cd build;
cmake ..
nice make -j
srun -N1 -n4 -c7 --gpu-bind=closest -t 05:00 ./GhostExchange -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 100
```

The output from this run should look like:

```
GhostExchange_ArrayAssign Timing is stencil 18.496334 boundary condition 0.020340 ghost cell 0.070608 total 19.140934
```

### Get a Trace

```
export HSA_XNACK=1
export OMNITRACE_CONFIG_FILE=~/.omnitrace.cfg
omnitrace-instrument -o ./GhostExchange.inst -- ./GhostExchange
srun -N1 -n4 -c7 --gpu-bind=closest -t 05:00 omnitrace-run -- ./GhostExchange.inst -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 100
```

The initial trace for this example should look very similar to previous traces we have seen:

![initial_trace](https://hackmd.io/_uploads/SJL25IJBR.png)

<!--![initial_trace](https://hackmd.io/_uploads/HyynqLoEC.png)-->


That is, we cannot see an obvious change in performance from just looking at this trace. We will make a note of the runtime of the Stencil kernel at line 167, its first and second instance took about 315 ms and 185 ms, respectively. In the next version we will compare these numbers to another modification.

### Look at Timemory output

We also see that our `wall_clock-0.txt` file looks pretty similar to our previous example:

![timemory_output_ver5](https://hackmd.io/_uploads/rJsGI_oVR.jpg)

To enable the output of this file, add `OMNITRACE_PROFILE=true` and `OMNITRACE_FLAT_PROFILE=true` to your `~/.omnitrace.cfg` file.

## Ghost Exchange Version 6: Explicit Memory Management

In this example we explicitly manage the memory movement onto the device by using `omp target enter data map`, `omp target update from`, and `omp target exit data map`, etc. instead of requiring OpenMP to use managed memory and have the OS manage page migrations automatically. We no longer need either the pragma `omp requires unified_shared_memory` on each translation unit or the `HSA_XNACK=1` setting.

Typically, startup costs of an application are not as important as the kernel runtimes. In this case, by explicitly moving memory at the beginning of our run, we're able to remove the overhead of memory movement from kernels. However our startup is slightly slower since we need to allocate a copy of all buffers on the device up-front.

### Build and Run

```
cd Ver6
mkdir build; cd build;
cmake ..
make -j8
srun -N1 -n4 -c7 --gpu-bind=closest -t 05:00 ./GhostExchange -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 100
```

The output for this run should look like:

```
GhostExchange_ArrayAssign Timing is stencil 0.141179 boundary condition 0.003179 ghost cell 0.807224 total 1.263204
```

Now we see a drastic improvement in our runtime. This should not be taken as typical for this type of optimization, as we saw a speedup of around 2x on a different system between the last example and this one. This points to the fact that this change avoids the overheads we were seeing due to some OpenMP configuration detail in previous examples.

### Get a Trace

```
unset HSA_XNACK
export OMNITRACE_CONFIG_FILE=~/.omnitrace.cfg
omnitrace-instrument -o ./GhostExchange.inst -- ./GhostExchange
srun -N1 -n4 -c7 --gpu-bind=closest -t 05:00 omnitrace-run -- ./GhostExchange.inst -x 2  -y 2  -i 20000 -j 20000 -h 2 -t -c -I 100
```

Here's what the trace looks like for this run:

![initial_trace](https://hackmd.io/_uploads/SkXeJP1HC.png)

<!--![initial_trace](https://hackmd.io/_uploads/SkN5o8oNA.png)-->


We see that BufAlloc seems to take much longer, but our kernels are much faster than before:
![zoomed_in](https://hackmd.io/_uploads/rk-oj8iNR.png)


We see here the same kernel we considered before, now at line 173, takes about 1.3ms! The implicit data movement was a large portion of our kernel overhead.

### Look at Timemory output

The `wall_clock-0.txt` file shows our overall run got much faster:

![timemory_output_ver6](https://hackmd.io/_uploads/ByddIui40.jpg)


Previously we ran in about 20 seconds, and now the uninstrumented runtime is around 1 seconds (from above), while `wall_clock-0.txt` shows our runtime is 2.3 seconds. However, we expect we should see a much more modest speedup, on the order of 2x. The exaggerated speedup is due to our initial GPU examples running more slowly than expected.

However, we see that the location of our data on CPU+GPU system matters quite a lot to performance. Implicit memory movement may not get the best performance, and it is usually worth it to pay the memory movement cost up front once than repeatedly for each kernel.
