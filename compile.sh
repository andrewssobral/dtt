export LIBTORCH_HOME=/path/to/libtorch
rm -rf build
cmake -S . -B build -DCMAKE_PREFIX_PATH=$LIBTORCH_HOME
cmake --build build --config Release
./build/dtt_test

# cmake -DCMAKE_PREFIX_PATH=$LIBTORCH_HOME .. -G Xcode
# open dtt.xcodeproj
