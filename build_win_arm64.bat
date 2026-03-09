@echo off
REM Build ExecuTorch for Windows ARM64 with XNNPACK CPU backend
REM Must run from Developer Command Prompt or with vcvars set

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" arm64

echo ===== Environment Ready =====
echo CL:
where cl
echo CMAKE:
where cmake
echo.

cd /d "C:\Users\100593\OmniInfer-LLM"

echo ===== Configuring CMake =====
cmake --preset windows -Bcmake-out-win-arm64 ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DEXECUTORCH_BUILD_QNN=OFF ^
  -DPYTHON_EXECUTABLE=C:\Python312\python.exe

if %ERRORLEVEL% NEQ 0 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

echo ===== Building =====
cmake --build cmake-out-win-arm64 --config Release -j12

if %ERRORLEVEL% NEQ 0 (
    echo Build failed!
    pause
    exit /b 1
)

echo ===== Build Complete =====
dir cmake-out-win-arm64\Release\*.exe 2>nul
dir cmake-out-win-arm64\*.exe 2>nul
pause
