/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#if defined(_WIN32)
#include <windows.h>
using LibHandle = HMODULE;
inline LibHandle LoadLib(const char* path) { return LoadLibraryA(path); }
inline void* GetSymbol(LibHandle handle, const char* name) { return reinterpret_cast<void*>(GetProcAddress(handle, name)); }
inline int CloseLib(LibHandle handle) { return FreeLibrary(handle) ? 0 : -1; }
inline const char* LibError() {
  static thread_local char buf[256];
  FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM, nullptr, GetLastError(), 0, buf, sizeof(buf), nullptr);
  return buf;
}
#else
#include <dlfcn.h>
using LibHandle = void*;
inline LibHandle LoadLib(const char* path) { return dlopen(path, RTLD_NOW | RTLD_LOCAL); }
inline void* GetSymbol(LibHandle handle, const char* name) { return dlsym(handle, name); }
inline int CloseLib(LibHandle handle) { return dlclose(handle); }
inline const char* LibError() { return dlerror(); }
#endif
