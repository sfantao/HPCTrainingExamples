#!/bin/bash

# This test checks that Omnitrace
# is able to produce .proto files
# NOTE: it does not check whether
# what is in those files is correct

module purge

module load rocm
REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
pushd ${REPO_DIR}/HIP/Stream_Overlap/0-Orig/
mkdir build_for_test; cd build_for_test
cmake ../
make -j

ROCM_VERSION=`cat ${ROCM_PATH}/.info/version | head -1 | cut -f1 -d'-' `
result=`echo ${ROCM_VERSION} | awk '$1<=6.1.2'` && echo $result
module unload rocm

if [[ "${result}" ]]; then
   module load omnitrace
   echo "loaded omnitrace from AMD Research"
   echo " "
else
   module load rocm
   echo "loaded omnitrace from ROCm"
   echo " "
fi  

omnitrace-avail -G $PWD/.omnitrace.cfg
export OMNITRACE_CONFIG_FILE=$PWD/.omnitrace.cfg
omnitrace-instrument -o compute_comm_overlap.inst -- compute_comm_overlap
omnitrace-run -- ./compute_comm_overlap.inst 2
cd omnitrace-compute_comm_overlap.inst-output/
ls *

cd ..
rm -rf build_for_test

popd

