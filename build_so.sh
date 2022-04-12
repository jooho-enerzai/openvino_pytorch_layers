cd user_ie_extensions
if [ -d  "build" ]
then
    cd build
else
    mkdir -p build
    cd build
fi
rm -rf *
/tmp/cmake-3.20.2-linux-x86_64/bin/cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4 VERBOSE=1
