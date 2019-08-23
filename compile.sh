rm -rf build && mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH=$LIBTORCH_HOME .. && make
./dtt_test
#cmake -DCMAKE_PREFIX_PATH=$LIBTORCH_HOME .. -G Xcode
#open dtt.xcodeproj
