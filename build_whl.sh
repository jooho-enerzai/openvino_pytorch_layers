rm -f *whl
rm -rf build
rm -rf dist
export VERSION=0
EXT_LIB=user_ie_extensions/build/libuser_cpu_extension.so python3 setup.py build bdist_wheel
mv dist/*.whl openvino_extensions-${VERSION}-py3-none-manylinux2014_x86_64.whl
if [ -d "openvino_extensions" ]
then
    cp user_ie_extensions/build/libuser_cpu_extension.so openvino_extensions/
fi
unset VERSION
