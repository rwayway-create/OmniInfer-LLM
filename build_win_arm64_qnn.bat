@echo off
REM Build ExecuTorch for Windows ARM64 with XNNPACK + QNN NPU backend
REM Must run from Developer Command Prompt or with vcvars set

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" arm64

echo ===== Environment Ready =====
echo CL:
where cl
echo CMAKE:
where cmake
echo.

set QNN_SDK_ROOT=C:\Qualcomm\AIStack\QAIRT\2.44.0.260225

cd /d "C:\Users\100593\OmniInfer-LLM"

echo ===== Configuring CMake (XNNPACK + QNN) =====
cmake --preset windows -Bcmake-out-qnn ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DEXECUTORCH_BUILD_QNN=ON ^
  -DQNN_SDK_ROOT=%QNN_SDK_ROOT% ^
  -DPYTHON_EXECUTABLE=C:\Python312\python.exe

if %ERRORLEVEL% NEQ 0 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

echo ===== Building =====
cmake --build cmake-out-qnn --config Release -j12

if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    pause
    exit /b 1
)

echo ===== Build Complete =====
echo.
echo Executables:
dir cmake-out-qnn\Release\*.exe 2>nul
dir cmake-out-qnn\*.exe 2>nul
echo.
echo QNN Libraries:
dir cmake-out-qnn\backends\qualcomm\*.dll 2>nul
dir cmake-out-qnn\backends\qualcomm\*.lib 2>nul
echo.
echo To run with QNN NPU, copy QNN DLLs to the exe directory:
echo   copy "%QNN_SDK_ROOT%\lib\aarch64-windows-msvc\QnnHtp.dll" cmake-out-qnn\
echo   copy "%QNN_SDK_ROOT%\lib\aarch64-windows-msvc\QnnHtpV73Stub.dll" cmake-out-qnn\
echo   copy "%QNN_SDK_ROOT%\lib\aarch64-windows-msvc\QnnSystem.dll" cmake-out-qnn\
pause
