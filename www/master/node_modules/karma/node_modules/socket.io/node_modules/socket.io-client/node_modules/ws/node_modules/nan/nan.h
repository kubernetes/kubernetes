/**********************************************************************************
 * NAN - Native Abstractions for Node.js
 *
 * Copyright (c) 2014 NAN contributors:
 *   - Rod Vagg <https://github.com/rvagg>
 *   - Benjamin Byholm <https://github.com/kkoopa>
 *   - Trevor Norris <https://github.com/trevnorris>
 *   - Nathan Rajlich <https://github.com/TooTallNate>
 *   - Brett Lawson <https://github.com/brett19>
 *   - Ben Noordhuis <https://github.com/bnoordhuis>
 *
 * MIT +no-false-attribs License <https://github.com/rvagg/nan/blob/master/LICENSE>
 *
 * Version 1.0.0 (current Node unstable: 0.11.13, Node stable: 0.10.28)
 *
 * ChangeLog:
 *  * 1.0.0 May 4 2014
 *    - Heavy API changes for V8 3.25 / Node 0.11.13
 *    - Use cpplint.py
 *    - Removed NanInitPersistent
 *    - Removed NanPersistentToLocal
 *    - Removed NanFromV8String
 *    - Removed NanMakeWeak
 *    - Removed NanNewLocal
 *    - Removed NAN_WEAK_CALLBACK_OBJECT
 *    - Removed NAN_WEAK_CALLBACK_DATA
 *    - Introduce NanNew, replaces NanNewLocal, NanPersistentToLocal, adds many overloaded typed versions
 *    - Introduce NanUndefined, NanNull, NanTrue and NanFalse
 *    - Introduce NanEscapableScope and NanEscapeScope
 *    - Introduce NanMakeWeakPersistent (requires a special callback to work on both old and new node)
 *    - Introduce NanMakeCallback for node::MakeCallback
 *    - Introduce NanSetTemplate
 *    - Introduce NanGetCurrentContext
 *    - Introduce NanCompileScript and NanRunScript
 *    - Introduce NanAdjustExternalMemory
 *    - Introduce NanAddGCEpilogueCallback, NanAddGCPrologueCallback, NanRemoveGCEpilogueCallback, NanRemoveGCPrologueCallback
 *    - Introduce NanGetHeapStatistics
 *    - Rename NanAsyncWorker#SavePersistent() to SaveToPersistent()
 *
 *  * 0.8.0 Jan 9 2014
 *    - NanDispose -> NanDisposePersistent, deprecate NanDispose
 *    - Extract _NAN_*_RETURN_TYPE, pull up NAN_*()
 *
 *  * 0.7.1 Jan 9 2014
 *    - Fixes to work against debug builds of Node
 *    - Safer NanPersistentToLocal (avoid reinterpret_cast)
 *    - Speed up common NanRawString case by only extracting flattened string when necessary
 *
 *  * 0.7.0 Dec 17 2013
 *    - New no-arg form of NanCallback() constructor.
 *    - NanCallback#Call takes Handle rather than Local
 *    - Removed deprecated NanCallback#Run method, use NanCallback#Call instead
 *    - Split off _NAN_*_ARGS_TYPE from _NAN_*_ARGS
 *    - Restore (unofficial) Node 0.6 compatibility at NanCallback#Call()
 *    - Introduce NanRawString() for char* (or appropriate void*) from v8::String
 *      (replacement for NanFromV8String)
 *    - Introduce NanCString() for null-terminated char* from v8::String
 *
 *  * 0.6.0 Nov 21 2013
 *    - Introduce NanNewLocal<T>(v8::Handle<T> value) for use in place of
 *      v8::Local<T>::New(...) since v8 started requiring isolate in Node 0.11.9
 *
 *  * 0.5.2 Nov 16 2013
 *    - Convert SavePersistent and GetFromPersistent in NanAsyncWorker from protected and public
 *
 *  * 0.5.1 Nov 12 2013
 *    - Use node::MakeCallback() instead of direct v8::Function::Call()
 *
 *  * 0.5.0 Nov 11 2013
 *    - Added @TooTallNate as collaborator
 *    - New, much simpler, "include_dirs" for binding.gyp
 *    - Added full range of NAN_INDEX_* macros to match NAN_PROPERTY_* macros
 *
 *  * 0.4.4 Nov 2 2013
 *    - Isolate argument from v8::Persistent::MakeWeak removed for 0.11.8+
 *
 *  * 0.4.3 Nov 2 2013
 *    - Include node_object_wrap.h, removed from node.h for Node 0.11.8.
 *
 *  * 0.4.2 Nov 2 2013
 *    - Handle deprecation of v8::Persistent::Dispose(v8::Isolate* isolate)) for
 *      Node 0.11.8 release.
 *
 *  * 0.4.1 Sep 16 2013
 *    - Added explicit `#include <uv.h>` as it was removed from node.h for v0.11.8
 *
 *  * 0.4.0 Sep 2 2013
 *    - Added NAN_INLINE and NAN_DEPRECATED and made use of them
 *    - Added NanError, NanTypeError and NanRangeError
 *    - Cleaned up code
 *
 *  * 0.3.2 Aug 30 2013
 *    - Fix missing scope declaration in GetFromPersistent() and SaveToPersistent
 *      in NanAsyncWorker
 *
 *  * 0.3.1 Aug 20 2013
 *    - fix "not all control paths return a value" compile warning on some platforms
 *
 *  * 0.3.0 Aug 19 2013
 *    - Made NAN work with NPM
 *    - Lots of fixes to NanFromV8String, pulling in features from new Node core
 *    - Changed node::encoding to Nan::Encoding in NanFromV8String to unify the API
 *    - Added optional error number argument for NanThrowError()
 *    - Added NanInitPersistent()
 *    - Added NanReturnNull() and NanReturnEmptyString()
 *    - Added NanLocker and NanUnlocker
 *    - Added missing scopes
 *    - Made sure to clear disposed Persistent handles
 *    - Changed NanAsyncWorker to allocate error messages on the heap
 *    - Changed NanThrowError(Local<Value>) to NanThrowError(Handle<Value>)
 *    - Fixed leak in NanAsyncWorker when errmsg is used
 *
 *  * 0.2.2 Aug 5 2013
 *    - Fixed usage of undefined variable with node::BASE64 in NanFromV8String()
 *
 *  * 0.2.1 Aug 5 2013
 *    - Fixed 0.8 breakage, node::BUFFER encoding type not available in 0.8 for
 *      NanFromV8String()
 *
 *  * 0.2.0 Aug 5 2013
 *    - Added NAN_PROPERTY_GETTER, NAN_PROPERTY_SETTER, NAN_PROPERTY_ENUMERATOR,
 *      NAN_PROPERTY_DELETER, NAN_PROPERTY_QUERY
 *    - Extracted _NAN_METHOD_ARGS, _NAN_GETTER_ARGS, _NAN_SETTER_ARGS,
 *      _NAN_PROPERTY_GETTER_ARGS, _NAN_PROPERTY_SETTER_ARGS,
 *      _NAN_PROPERTY_ENUMERATOR_ARGS, _NAN_PROPERTY_DELETER_ARGS,
 *      _NAN_PROPERTY_QUERY_ARGS
 *    - Added NanGetInternalFieldPointer, NanSetInternalFieldPointer
 *    - Added NAN_WEAK_CALLBACK, NAN_WEAK_CALLBACK_OBJECT,
 *      NAN_WEAK_CALLBACK_DATA, NanMakeWeak
 *    - Renamed THROW_ERROR to _NAN_THROW_ERROR
 *    - Added NanNewBufferHandle(char*, size_t, node::smalloc::FreeCallback, void*)
 *    - Added NanBufferUse(char*, uint32_t)
 *    - Added NanNewContextHandle(v8::ExtensionConfiguration*,
 *        v8::Handle<v8::ObjectTemplate>, v8::Handle<v8::Value>)
 *    - Fixed broken NanCallback#GetFunction()
 *    - Added optional encoding and size arguments to NanFromV8String()
 *    - Added NanGetPointerSafe() and NanSetPointerSafe()
 *    - Added initial test suite (to be expanded)
 *    - Allow NanUInt32OptionValue to convert any Number object
 *
 *  * 0.1.0 Jul 21 2013
 *    - Added `NAN_GETTER`, `NAN_SETTER`
 *    - Added `NanThrowError` with single Local<Value> argument
 *    - Added `NanNewBufferHandle` with single uint32_t argument
 *    - Added `NanHasInstance(Persistent<FunctionTemplate>&, Handle<Value>)`
 *    - Added `Local<Function> NanCallback#GetFunction()`
 *    - Added `NanCallback#Call(int, Local<Value>[])`
 *    - Deprecated `NanCallback#Run(int, Local<Value>[])` in favour of Call
 *
 * See https://github.com/rvagg/nan for the latest update to this file
 **********************************************************************************/

#ifndef NAN_H_
#define NAN_H_

#include <uv.h>
#include <node.h>
#include <node_buffer.h>
#include <node_version.h>
#include <node_object_wrap.h>
#include <string.h>

#if defined(__GNUC__) && !defined(DEBUG)
# define NAN_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER) && !defined(DEBUG)
# define NAN_INLINE __forceinline
#else
# define NAN_INLINE inline
#endif

#if defined(__GNUC__) && !V8_DISABLE_DEPRECATIONS
# define NAN_DEPRECATED __attribute__((deprecated))
#elif defined(_MSC_VER) && !V8_DISABLE_DEPRECATIONS
# define NAN_DEPRECATED __declspec(deprecated)
#else
# define NAN_DEPRECATED
#endif

// some generic helpers

template<typename T> NAN_INLINE bool NanSetPointerSafe(
    T *var
  , T val
) {
  if (var) {
    *var = val;
    return true;
  } else {
    return false;
  }
}

template<typename T> NAN_INLINE T NanGetPointerSafe(
    T *var
  , T fallback = reinterpret_cast<T>(0)
) {
  if (var) {
    return *var;
  } else {
    return fallback;
  }
}

NAN_INLINE bool NanBooleanOptionValue(
    v8::Local<v8::Object> optionsObj
  , v8::Handle<v8::String> opt, bool def
) {
  if (def) {
    return optionsObj.IsEmpty()
      || !optionsObj->Has(opt)
      || optionsObj->Get(opt)->BooleanValue();
  } else {
    return !optionsObj.IsEmpty()
      && optionsObj->Has(opt)
      && optionsObj->Get(opt)->BooleanValue();
  }
}

NAN_INLINE bool NanBooleanOptionValue(
    v8::Local<v8::Object> optionsObj
  , v8::Handle<v8::String> opt
) {
  return NanBooleanOptionValue(optionsObj, opt, false);
}

NAN_INLINE uint32_t NanUInt32OptionValue(
    v8::Local<v8::Object> optionsObj
  , v8::Handle<v8::String> opt
  , uint32_t def
) {
  return !optionsObj.IsEmpty()
    && optionsObj->Has(opt)
    && optionsObj->Get(opt)->IsNumber()
      ? optionsObj->Get(opt)->Uint32Value()
      : def;
}

#if (NODE_MODULE_VERSION > 0x000B)
// Node 0.11+ (0.11.3 and below won't compile with these)

# define _NAN_METHOD_ARGS_TYPE const v8::FunctionCallbackInfo<v8::Value>&
# define _NAN_METHOD_ARGS _NAN_METHOD_ARGS_TYPE args
# define _NAN_METHOD_RETURN_TYPE void

# define _NAN_GETTER_ARGS_TYPE const v8::PropertyCallbackInfo<v8::Value>&
# define _NAN_GETTER_ARGS _NAN_GETTER_ARGS_TYPE args
# define _NAN_GETTER_RETURN_TYPE void

# define _NAN_SETTER_ARGS_TYPE const v8::PropertyCallbackInfo<void>&
# define _NAN_SETTER_ARGS _NAN_SETTER_ARGS_TYPE args
# define _NAN_SETTER_RETURN_TYPE void

# define _NAN_PROPERTY_GETTER_ARGS_TYPE                                        \
    const v8::PropertyCallbackInfo<v8::Value>&
# define _NAN_PROPERTY_GETTER_ARGS _NAN_PROPERTY_GETTER_ARGS_TYPE args
# define _NAN_PROPERTY_GETTER_RETURN_TYPE void

# define _NAN_PROPERTY_SETTER_ARGS_TYPE                                        \
    const v8::PropertyCallbackInfo<v8::Value>&
# define _NAN_PROPERTY_SETTER_ARGS _NAN_PROPERTY_SETTER_ARGS_TYPE args
# define _NAN_PROPERTY_SETTER_RETURN_TYPE void

# define _NAN_PROPERTY_ENUMERATOR_ARGS_TYPE                                    \
    const v8::PropertyCallbackInfo<v8::Array>&
# define _NAN_PROPERTY_ENUMERATOR_ARGS _NAN_PROPERTY_ENUMERATOR_ARGS_TYPE args
# define _NAN_PROPERTY_ENUMERATOR_RETURN_TYPE void

# define _NAN_PROPERTY_DELETER_ARGS_TYPE                                       \
    const v8::PropertyCallbackInfo<v8::Boolean>&
# define _NAN_PROPERTY_DELETER_ARGS                                            \
    _NAN_PROPERTY_DELETER_ARGS_TYPE args
# define _NAN_PROPERTY_DELETER_RETURN_TYPE void

# define _NAN_PROPERTY_QUERY_ARGS_TYPE                                         \
    const v8::PropertyCallbackInfo<v8::Integer>&
# define _NAN_PROPERTY_QUERY_ARGS _NAN_PROPERTY_QUERY_ARGS_TYPE args
# define _NAN_PROPERTY_QUERY_RETURN_TYPE void

# define _NAN_INDEX_GETTER_ARGS_TYPE                                           \
    const v8::PropertyCallbackInfo<v8::Value>&
# define _NAN_INDEX_GETTER_ARGS _NAN_INDEX_GETTER_ARGS_TYPE args
# define _NAN_INDEX_GETTER_RETURN_TYPE void

# define _NAN_INDEX_SETTER_ARGS_TYPE                                           \
    const v8::PropertyCallbackInfo<v8::Value>&
# define _NAN_INDEX_SETTER_ARGS _NAN_INDEX_SETTER_ARGS_TYPE args
# define _NAN_INDEX_SETTER_RETURN_TYPE void

# define _NAN_INDEX_ENUMERATOR_ARGS_TYPE                                       \
    const v8::PropertyCallbackInfo<v8::Array>&
# define _NAN_INDEX_ENUMERATOR_ARGS _NAN_INDEX_ENUMERATOR_ARGS_TYPE args
# define _NAN_INDEX_ENUMERATOR_RETURN_TYPE void

# define _NAN_INDEX_DELETER_ARGS_TYPE                                          \
    const v8::PropertyCallbackInfo<v8::Boolean>&
# define _NAN_INDEX_DELETER_ARGS _NAN_INDEX_DELETER_ARGS_TYPE args
# define _NAN_INDEX_DELETER_RETURN_TYPE void

# define _NAN_INDEX_QUERY_ARGS_TYPE                                            \
    const v8::PropertyCallbackInfo<v8::Integer>&
# define _NAN_INDEX_QUERY_ARGS _NAN_INDEX_QUERY_ARGS_TYPE args
# define _NAN_INDEX_QUERY_RETURN_TYPE void

typedef v8::FunctionCallback NanFunctionCallback;
static v8::Isolate* nan_isolate = v8::Isolate::GetCurrent();

# define NanUndefined() v8::Undefined(nan_isolate)
# define NanNull() v8::Null(nan_isolate)
# define NanTrue() v8::True(nan_isolate)
# define NanFalse() v8::False(nan_isolate)
# define NanAdjustExternalMemory(amount)                                       \
    nan_isolate->AdjustAmountOfExternalAllocatedMemory(amount)
# define NanSetTemplate(templ, name, value) templ->Set(nan_isolate, name, value)
# define NanGetCurrentContext() nan_isolate->GetCurrentContext()
# define NanMakeCallback(target, func, argc, argv)                             \
    node::MakeCallback(nan_isolate, target, func, argc, argv)
# define NanGetInternalFieldPointer(object, index)                             \
    object->GetAlignedPointerFromInternalField(index)
# define NanSetInternalFieldPointer(object, index, value)                      \
    object->SetAlignedPointerInInternalField(index, value)

  template<typename T>
  NAN_INLINE v8::Local<T> NanNew() {
    return T::New(nan_isolate);
  }

  template<typename T, typename P>
  NAN_INLINE v8::Local<T> NanNew(P arg1) {
    return T::New(nan_isolate, arg1);
  }

  template<typename T>
  NAN_INLINE v8::Local<v8::Signature> NanNew(
      v8::Handle<v8::FunctionTemplate> receiver
    , int argc
    , v8::Handle<v8::FunctionTemplate> argv[] = 0) {
    return v8::Signature::New(nan_isolate, receiver, argc, argv);
  }

  template<typename T>
  NAN_INLINE v8::Local<v8::FunctionTemplate> NanNew(
      NanFunctionCallback callback
    , v8::Handle<v8::Value> data = v8::Handle<v8::Value>()
    , v8::Handle<v8::Signature> signature = v8::Handle<v8::Signature>()) {
    return T::New(nan_isolate, callback, data, signature);
  }

  template<typename T>
  NAN_INLINE v8::Local<T> NanNew(v8::Handle<T> arg1) {
    return v8::Local<T>::New(nan_isolate, arg1);
  }

  template<typename T>
  NAN_INLINE v8::Local<T> NanNew(const v8::Persistent<T> &arg1) {
    return v8::Local<T>::New(nan_isolate, arg1);
  }

  template<typename T, typename P>
  NAN_INLINE v8::Local<T> NanNew(P arg1, int arg2) {
    return T::New(nan_isolate, arg1, arg2);
  }

  template<>
  NAN_INLINE v8::Local<v8::Array> NanNew<v8::Array>() {
    return v8::Array::New(nan_isolate);
  }

  template<>
  NAN_INLINE v8::Local<v8::Array> NanNew<v8::Array>(int length) {
    return v8::Array::New(nan_isolate, length);
  }

  template<>
  NAN_INLINE v8::Local<v8::Date> NanNew<v8::Date>(double time) {
    return v8::Date::New(nan_isolate, time).As<v8::Date>();
  }

  template<>
  NAN_INLINE v8::Local<v8::Date> NanNew<v8::Date>(int time) {
    return v8::Date::New(nan_isolate, time).As<v8::Date>();
  }

  typedef v8::UnboundScript NanUnboundScript;
  typedef v8::Script NanBoundScript;

  template<typename T, typename P>
  NAN_INLINE v8::Local<T> NanNew(
      P s
    , const v8::ScriptOrigin& origin
  ) {
    v8::ScriptCompiler::Source source(s, origin);
    return v8::ScriptCompiler::CompileUnbound(nan_isolate, &source);
  }

  template<>
  NAN_INLINE v8::Local<NanUnboundScript> NanNew<NanUnboundScript>(
      v8::Local<v8::String> s
  ) {
    v8::ScriptCompiler::Source source(s);
    return v8::ScriptCompiler::CompileUnbound(nan_isolate, &source);
  }

  NAN_INLINE v8::Local<v8::String> NanNew(
      v8::String::ExternalStringResource *resource) {
    return v8::String::NewExternal(nan_isolate, resource);
  }

  NAN_INLINE v8::Local<v8::String> NanNew(
      v8::String::ExternalAsciiStringResource *resource) {
    return v8::String::NewExternal(nan_isolate, resource);
  }

  template<>
  NAN_INLINE v8::Local<v8::BooleanObject> NanNew(bool value) {
    return v8::BooleanObject::New(value).As<v8::BooleanObject>();
  }

  template<>
  NAN_INLINE v8::Local<v8::StringObject>
  NanNew<v8::StringObject, v8::Local<v8::String> >(
      v8::Local<v8::String> value) {
    return v8::StringObject::New(value).As<v8::StringObject>();
  }

  template<>
  NAN_INLINE v8::Local<v8::StringObject>
  NanNew<v8::StringObject, v8::Handle<v8::String> >(
      v8::Handle<v8::String> value) {
    return v8::StringObject::New(value).As<v8::StringObject>();
  }

  template<>
  NAN_INLINE v8::Local<v8::NumberObject> NanNew<v8::NumberObject>(double val) {
    return v8::NumberObject::New(nan_isolate, val).As<v8::NumberObject>();
  }

  template<typename T>
  NAN_INLINE v8::Local<v8::RegExp> NanNew(
      v8::Handle<v8::String> pattern, v8::RegExp::Flags flags) {
    return v8::RegExp::New(pattern, flags);
  }

  template<typename T>
  NAN_INLINE v8::Local<v8::RegExp> NanNew(
      v8::Local<v8::String> pattern, v8::RegExp::Flags flags) {
    return v8::RegExp::New(pattern, flags);
  }

  template<typename T, typename P>
  NAN_INLINE v8::Local<v8::RegExp> NanNew(
      v8::Handle<v8::String> pattern, v8::RegExp::Flags flags) {
    return v8::RegExp::New(pattern, flags);
  }

  template<typename T, typename P>
  NAN_INLINE v8::Local<v8::RegExp> NanNew(
      v8::Local<v8::String> pattern, v8::RegExp::Flags flags) {
    return v8::RegExp::New(pattern, flags);
  }

  template<>
  NAN_INLINE v8::Local<v8::Uint32> NanNew<v8::Uint32, int32_t>(int32_t val) {
    return v8::Uint32::NewFromUnsigned(nan_isolate, val)->ToUint32();
  }

  template<>
  NAN_INLINE v8::Local<v8::Uint32> NanNew<v8::Uint32, uint32_t>(uint32_t val) {
    return v8::Uint32::NewFromUnsigned(nan_isolate, val)->ToUint32();
  }

  template<>
  NAN_INLINE v8::Local<v8::Int32> NanNew<v8::Int32, int32_t>(int32_t val) {
    return v8::Int32::New(nan_isolate, val)->ToInt32();
  }

  template<>
  NAN_INLINE v8::Local<v8::Int32> NanNew<v8::Int32, uint32_t>(uint32_t val) {
    return v8::Int32::New(nan_isolate, val)->ToInt32();
  }

  template<>
  NAN_INLINE v8::Local<v8::String> NanNew<v8::String, char *>(
      char *arg
    , int length) {
    return v8::String::NewFromUtf8(
        nan_isolate
      , arg
      , v8::String::kNormalString
      , length);
  }

  template<>
  NAN_INLINE v8::Local<v8::String> NanNew<v8::String, const char *>(
      const char *arg
    , int length) {
    return v8::String::NewFromUtf8(
        nan_isolate
      , arg
      , v8::String::kNormalString
      , length);
  }

  template<>
  NAN_INLINE v8::Local<v8::String> NanNew<v8::String, char *>(char *arg) {
    return v8::String::NewFromUtf8(nan_isolate, arg);
  }

  template<>
  NAN_INLINE v8::Local<v8::String> NanNew<v8::String, const char *>(
      const char *arg) {
    return v8::String::NewFromUtf8(nan_isolate, arg);
  }

  template<>
  NAN_INLINE v8::Local<v8::String> NanNew<v8::String, uint8_t *>(
      uint8_t *arg
    , int length) {
    return v8::String::NewFromOneByte(
        nan_isolate
      , arg
      , v8::String::kNormalString
      , length);
  }

  template<>
  NAN_INLINE v8::Local<v8::String> NanNew<v8::String, const uint8_t *>(
      const uint8_t *arg
    , int length) {
    return v8::String::NewFromOneByte(
        nan_isolate
      , arg
      , v8::String::kNormalString
      , length);
  }

  template<>
  NAN_INLINE v8::Local<v8::String> NanNew<v8::String, uint8_t *>(uint8_t *arg) {
    return v8::String::NewFromOneByte(nan_isolate, arg);
  }

  template<>
  NAN_INLINE v8::Local<v8::String> NanNew<v8::String, const uint8_t *>(
      const uint8_t *arg) {
    return v8::String::NewFromOneByte(nan_isolate, arg);
  }

  template<>
  NAN_INLINE v8::Local<v8::String> NanNew<v8::String, uint16_t *>(
      uint16_t *arg
    , int length) {
    return v8::String::NewFromTwoByte(
        nan_isolate
      , arg
      , v8::String::kNormalString
      , length);
  }

  template<>
  NAN_INLINE v8::Local<v8::String> NanNew<v8::String, const uint16_t *>(
      const uint16_t *arg
    , int length) {
    return v8::String::NewFromTwoByte(
        nan_isolate
      , arg
      , v8::String::kNormalString
      , length);
  }
  template<>
  NAN_INLINE v8::Local<v8::String> NanNew<v8::String, uint16_t *>(
      uint16_t *arg) {
    return v8::String::NewFromTwoByte(nan_isolate, arg);
  }

  template<>
  NAN_INLINE v8::Local<v8::String> NanNew<v8::String, const uint16_t *>(
      const uint16_t *arg) {
    return v8::String::NewFromTwoByte(nan_isolate, arg);
  }

  template<>
  NAN_INLINE v8::Local<v8::String> NanNew<v8::String>() {
    return v8::String::Empty(nan_isolate);
  }

  NAN_INLINE void NanAddGCEpilogueCallback(
      v8::Isolate::GCEpilogueCallback callback
    , v8::GCType gc_type_filter = v8::kGCTypeAll) {
    nan_isolate->AddGCEpilogueCallback(callback, gc_type_filter);
  }

  NAN_INLINE void NanRemoveGCEpilogueCallback(
      v8::Isolate::GCEpilogueCallback callback) {
    nan_isolate->RemoveGCEpilogueCallback(callback);
  }

  NAN_INLINE void NanAddGCPrologueCallback(
      v8::Isolate::GCPrologueCallback callback
    , v8::GCType gc_type_filter = v8::kGCTypeAll) {
    nan_isolate->AddGCPrologueCallback(callback, gc_type_filter);
  }

  NAN_INLINE void NanRemoveGCPrologueCallback(
      v8::Isolate::GCPrologueCallback callback) {
    nan_isolate->RemoveGCPrologueCallback(callback);
  }

  NAN_INLINE void NanGetHeapStatistics(
      v8::HeapStatistics *heap_statistics) {
    nan_isolate->GetHeapStatistics(heap_statistics);
  }

# define NanSymbol(value) NanNew<v8::String>(value)

  template<typename T>
  NAN_INLINE void NanAssignPersistent(
      v8::Persistent<T>& handle
    , v8::Handle<T> obj) {
      handle.Reset(nan_isolate, obj);
  }

  template<typename T>
  NAN_INLINE void NanAssignPersistent(
      v8::Persistent<T>& handle
    , const v8::Persistent<T>& obj) {
      handle.Reset(nan_isolate, obj);
  }

  template<typename T, typename P>
  struct _NanWeakCallbackInfo {
    typedef void (*Callback)(
      const v8::WeakCallbackData<T, _NanWeakCallbackInfo<T, P> >& data);
    _NanWeakCallbackInfo(v8::Handle<T> handle, P* param, Callback cb)
      : parameter(param), callback(cb) {
       NanAssignPersistent(persistent, handle);
    }

    ~_NanWeakCallbackInfo() {
      persistent.Reset();
    }

    P* const parameter;
    Callback const callback;
    v8::Persistent<T> persistent;
  };

  template<typename T, typename P>
  class _NanWeakCallbackData {
   public:
    _NanWeakCallbackData(_NanWeakCallbackInfo<T, P> *info)
      : info_(info) { }

    NAN_INLINE v8::Local<T> GetValue() const {
      return NanNew(info_->persistent);
    }
    NAN_INLINE P* GetParameter() const { return info_->parameter; }
    NAN_INLINE void Revive() const {
      info_->persistent.SetWeak(info_, info_->callback);
    }

    NAN_INLINE void Dispose() const {
      delete info_;
    }

   private:
    _NanWeakCallbackInfo<T, P>* info_;
  };

// do not use for declaration
# define NAN_WEAK_CALLBACK(name)                                               \
    template<typename T, typename P>                                           \
    static void name(                                                          \
      const v8::WeakCallbackData<T, _NanWeakCallbackInfo<T, P> > &data) {      \
        _NanWeakCallbackData<T, P> wcbd(                                       \
           data.GetParameter());                                               \
        _Nan_Weak_Callback_ ## name(wcbd);                                     \
    }                                                                          \
                                                                               \
    template<typename T, typename P>                                           \
    NAN_INLINE void _Nan_Weak_Callback_ ## name(                               \
        const _NanWeakCallbackData<T, P> &data)

# define NanScope() v8::HandleScope scope(nan_isolate)
# define NanEscapableScope() v8::EscapableHandleScope scope(nan_isolate)
# define NanEscapeScope(val) scope.Escape(val)
# define NanLocker() v8::Locker locker(nan_isolate)
# define NanUnlocker() v8::Unlocker unlocker(nan_isolate)
# define NanReturnValue(value) return args.GetReturnValue().Set(value)
# define NanReturnUndefined() return
# define NanReturnNull() return args.GetReturnValue().SetNull()
# define NanReturnEmptyString() return args.GetReturnValue().SetEmptyString()

# define NanObjectWrapHandle(obj) obj->handle()

template<typename T, typename P>
void NAN_INLINE NanMakeWeakPersistent(
    v8::Handle<T> handle
  , P* parameter
  , typename _NanWeakCallbackInfo<T, P>::Callback callback) {
    _NanWeakCallbackInfo<T, P> *cbinfo =
     new _NanWeakCallbackInfo<T, P>(handle, parameter, callback);
    cbinfo->persistent.SetWeak(cbinfo, callback);
}

# define _NAN_ERROR(fun, errmsg) fun(NanNew<v8::String>(errmsg))

# define _NAN_THROW_ERROR(fun, errmsg)                                         \
    do {                                                                       \
      NanScope();                                                              \
      nan_isolate->ThrowException(_NAN_ERROR(fun, errmsg));                    \
    } while (0);

  NAN_INLINE v8::Local<v8::Value> NanError(const char* errmsg) {
    return  _NAN_ERROR(v8::Exception::Error, errmsg);
  }

  NAN_INLINE void NanThrowError(const char* errmsg) {
    _NAN_THROW_ERROR(v8::Exception::Error, errmsg);
  }

  NAN_INLINE void NanThrowError(v8::Handle<v8::Value> error) {
    NanScope();
    nan_isolate->ThrowException(error);
  }

  NAN_INLINE v8::Local<v8::Value> NanError(
      const char *msg
    , const int errorNumber
  ) {
    v8::Local<v8::Value> err = v8::Exception::Error(NanNew<v8::String>(msg));
    v8::Local<v8::Object> obj = err.As<v8::Object>();
    obj->Set(NanSymbol("code"), NanNew<v8::Integer>(errorNumber));
    return err;
  }

  NAN_INLINE void NanThrowError(
      const char *msg
    , const int errorNumber
  ) {
    NanThrowError(NanError(msg, errorNumber));
  }

  NAN_INLINE v8::Local<v8::Value> NanTypeError(const char* errmsg) {
    return _NAN_ERROR(v8::Exception::TypeError, errmsg);
  }

  NAN_INLINE void NanThrowTypeError(const char* errmsg) {
    _NAN_THROW_ERROR(v8::Exception::TypeError, errmsg);
  }

  NAN_INLINE v8::Local<v8::Value> NanRangeError(const char* errmsg) {
    return _NAN_ERROR(v8::Exception::RangeError, errmsg);
  }

  NAN_INLINE void NanThrowRangeError(const char* errmsg) {
    _NAN_THROW_ERROR(v8::Exception::RangeError, errmsg);
  }

  template<typename T> NAN_INLINE void NanDisposePersistent(
      v8::Persistent<T> &handle
  ) {
    handle.Reset();
  }

  NAN_INLINE v8::Local<v8::Object> NanNewBufferHandle (
      char *data
    , size_t length
    , node::smalloc::FreeCallback callback
    , void *hint
  ) {
    return node::Buffer::New(nan_isolate, data, length, callback, hint);
  }

  NAN_INLINE v8::Local<v8::Object> NanNewBufferHandle (
      const char *data
    , uint32_t size
  ) {
    return node::Buffer::New(nan_isolate, data, size);
  }

  NAN_INLINE v8::Local<v8::Object> NanNewBufferHandle (uint32_t size) {
    return node::Buffer::New(nan_isolate, size);
  }

  NAN_INLINE v8::Local<v8::Object> NanBufferUse(
      char* data
    , uint32_t size
  ) {
    return node::Buffer::Use(nan_isolate, data, size);
  }

  NAN_INLINE bool NanHasInstance(
      v8::Persistent<v8::FunctionTemplate>& function_template
    , v8::Handle<v8::Value> value
  ) {
    return NanNew(function_template)->HasInstance(value);
  }

  NAN_INLINE v8::Local<v8::Context> NanNewContextHandle(
      v8::ExtensionConfiguration* extensions = NULL
    , v8::Handle<v8::ObjectTemplate> tmpl = v8::Handle<v8::ObjectTemplate>()
    , v8::Handle<v8::Value> obj = v8::Handle<v8::Value>()
  ) {
    return v8::Local<v8::Context>::New(
        nan_isolate
      , v8::Context::New(nan_isolate, extensions, tmpl, obj)
    );
  }

  NAN_INLINE v8::Local<NanBoundScript> NanCompileScript(
      v8::Local<v8::String> s
    , const v8::ScriptOrigin& origin
  ) {
    v8::ScriptCompiler::Source source(s, origin);
    return v8::ScriptCompiler::Compile(nan_isolate, &source);
  }

  NAN_INLINE v8::Local<NanBoundScript> NanCompileScript(
      v8::Local<v8::String> s
  ) {
    v8::ScriptCompiler::Source source(s);
    return v8::ScriptCompiler::Compile(nan_isolate, &source);
  }

  NAN_INLINE v8::Local<v8::Value> NanRunScript(
      v8::Local<NanUnboundScript> script
  ) {
    return script->BindToCurrentContext()->Run();
  }

  NAN_INLINE v8::Local<v8::Value> NanRunScript(
      v8::Local<NanBoundScript> script
  ) {
    return script->Run();
  }

#else
// Node 0.8 and 0.10

# define _NAN_METHOD_ARGS_TYPE const v8::Arguments&
# define _NAN_METHOD_ARGS _NAN_METHOD_ARGS_TYPE args
# define _NAN_METHOD_RETURN_TYPE v8::Handle<v8::Value>

# define _NAN_GETTER_ARGS_TYPE const v8::AccessorInfo &
# define _NAN_GETTER_ARGS _NAN_GETTER_ARGS_TYPE args
# define _NAN_GETTER_RETURN_TYPE v8::Handle<v8::Value>

# define _NAN_SETTER_ARGS_TYPE const v8::AccessorInfo &
# define _NAN_SETTER_ARGS _NAN_SETTER_ARGS_TYPE args
# define _NAN_SETTER_RETURN_TYPE void

# define _NAN_PROPERTY_GETTER_ARGS_TYPE const v8::AccessorInfo&
# define _NAN_PROPERTY_GETTER_ARGS _NAN_PROPERTY_GETTER_ARGS_TYPE args
# define _NAN_PROPERTY_GETTER_RETURN_TYPE v8::Handle<v8::Value>

# define _NAN_PROPERTY_SETTER_ARGS_TYPE const v8::AccessorInfo&
# define _NAN_PROPERTY_SETTER_ARGS _NAN_PROPERTY_SETTER_ARGS_TYPE args
# define _NAN_PROPERTY_SETTER_RETURN_TYPE v8::Handle<v8::Value>

# define _NAN_PROPERTY_ENUMERATOR_ARGS_TYPE const v8::AccessorInfo&
# define _NAN_PROPERTY_ENUMERATOR_ARGS _NAN_PROPERTY_ENUMERATOR_ARGS_TYPE args
# define _NAN_PROPERTY_ENUMERATOR_RETURN_TYPE v8::Handle<v8::Array>

# define _NAN_PROPERTY_DELETER_ARGS_TYPE const v8::AccessorInfo&
# define _NAN_PROPERTY_DELETER_ARGS _NAN_PROPERTY_DELETER_ARGS_TYPE args
# define _NAN_PROPERTY_DELETER_RETURN_TYPE v8::Handle<v8::Boolean>

# define _NAN_PROPERTY_QUERY_ARGS_TYPE const v8::AccessorInfo&
# define _NAN_PROPERTY_QUERY_ARGS _NAN_PROPERTY_QUERY_ARGS_TYPE args
# define _NAN_PROPERTY_QUERY_RETURN_TYPE v8::Handle<v8::Integer>

# define _NAN_INDEX_GETTER_ARGS_TYPE const v8::AccessorInfo&
# define _NAN_INDEX_GETTER_ARGS _NAN_INDEX_GETTER_ARGS_TYPE args
# define _NAN_INDEX_GETTER_RETURN_TYPE v8::Handle<v8::Value>

# define _NAN_INDEX_SETTER_ARGS_TYPE const v8::AccessorInfo&
# define _NAN_INDEX_SETTER_ARGS _NAN_INDEX_SETTER_ARGS_TYPE args
# define _NAN_INDEX_SETTER_RETURN_TYPE v8::Handle<v8::Value>

# define _NAN_INDEX_ENUMERATOR_ARGS_TYPE const v8::AccessorInfo&
# define _NAN_INDEX_ENUMERATOR_ARGS _NAN_INDEX_ENUMERATOR_ARGS_TYPE args
# define _NAN_INDEX_ENUMERATOR_RETURN_TYPE v8::Handle<v8::Array>

# define _NAN_INDEX_DELETER_ARGS_TYPE const v8::AccessorInfo&
# define _NAN_INDEX_DELETER_ARGS _NAN_INDEX_DELETER_ARGS_TYPE args
# define _NAN_INDEX_DELETER_RETURN_TYPE v8::Handle<v8::Boolean>

# define _NAN_INDEX_QUERY_ARGS_TYPE const v8::AccessorInfo&
# define _NAN_INDEX_QUERY_ARGS _NAN_INDEX_QUERY_ARGS_TYPE args
# define _NAN_INDEX_QUERY_RETURN_TYPE v8::Handle<v8::Integer>

typedef v8::InvocationCallback NanFunctionCallback;

# define NanUndefined() v8::Undefined()
# define NanNull() v8::Null()
# define NanTrue() v8::True()
# define NanFalse() v8::False()
# define NanAdjustExternalMemory(amount)                                       \
    v8::V8::AdjustAmountOfExternalAllocatedMemory(amount)
# define NanSetTemplate(templ, name, value) templ->Set(name, value)
# define NanGetCurrentContext() v8::Context::GetCurrent()
# if NODE_VERSION_AT_LEAST(0, 8, 0)
#  define NanMakeCallback(target, func, argc, argv)                            \
    node::MakeCallback(target, func, argc, argv)
# else
#  define NanMakeCallback(target, func, argc, argv)                            \
    do {                                                                       \
      v8::TryCatch try_catch;                                                  \
      func->Call(target, argc, argv);                                          \
      if (try_catch.HasCaught()) {                                             \
          v8::FatalException(try_catch);                                       \
      }                                                                        \
    } while (0)
# endif

# define NanSymbol(value) v8::String::NewSymbol(value)

  template<typename T>
  NAN_INLINE v8::Local<T> NanNew() {
    return v8::Local<T>::New(T::New());
  }

  template<typename T>
  NAN_INLINE v8::Local<T> NanNew(v8::Handle<T> arg) {
    return v8::Local<T>::New(arg);
  }

  template<typename T>
  NAN_INLINE v8::Local<v8::Signature> NanNew(
      v8::Handle<v8::FunctionTemplate> receiver
    , int argc
    , v8::Handle<v8::FunctionTemplate> argv[] = 0) {
    return v8::Signature::New(receiver, argc, argv);
  }

  template<typename T>
  NAN_INLINE v8::Local<v8::FunctionTemplate> NanNew(
      NanFunctionCallback callback
    , v8::Handle<v8::Value> data = v8::Handle<v8::Value>()
    , v8::Handle<v8::Signature> signature = v8::Handle<v8::Signature>()) {
    return T::New(callback, data, signature);
  }

  template<typename T>
  NAN_INLINE v8::Local<T> NanNew(const v8::Persistent<T> &arg) {
    return v8::Local<T>::New(arg);
  }

  template<typename T, typename P>
  NAN_INLINE v8::Local<T> NanNew(P arg) {
    return v8::Local<T>::New(T::New(arg));
  }

  template<typename T, typename P>
  NAN_INLINE v8::Local<T> NanNew(P arg, int length) {
    return v8::Local<T>::New(T::New(arg, length));
  }

  template<typename T>
  NAN_INLINE v8::Local<v8::RegExp> NanNew(
      v8::Handle<v8::String> pattern, v8::RegExp::Flags flags) {
    return v8::RegExp::New(pattern, flags);
  }

  template<typename T>
  NAN_INLINE v8::Local<v8::RegExp> NanNew(
      v8::Local<v8::String> pattern, v8::RegExp::Flags flags) {
    return v8::RegExp::New(pattern, flags);
  }

  template<typename T, typename P>
  NAN_INLINE v8::Local<v8::RegExp> NanNew(
      v8::Handle<v8::String> pattern, v8::RegExp::Flags flags) {
    return v8::RegExp::New(pattern, flags);
  }

  template<typename T, typename P>
  NAN_INLINE v8::Local<v8::RegExp> NanNew(
      v8::Local<v8::String> pattern, v8::RegExp::Flags flags) {
    return v8::RegExp::New(pattern, flags);
  }

  template<>
  NAN_INLINE v8::Local<v8::Array> NanNew<v8::Array>() {
    return v8::Array::New();
  }

  template<>
  NAN_INLINE v8::Local<v8::Array> NanNew<v8::Array>(int length) {
    return v8::Array::New(length);
  }


  template<>
  NAN_INLINE v8::Local<v8::Date> NanNew<v8::Date>(double time) {
    return v8::Date::New(time).As<v8::Date>();
  }

  template<>
  NAN_INLINE v8::Local<v8::Date> NanNew<v8::Date>(int time) {
    return v8::Date::New(time).As<v8::Date>();
  }

  typedef v8::Script NanUnboundScript;
  typedef v8::Script NanBoundScript;

  template<typename T, typename P>
  NAN_INLINE v8::Local<T> NanNew(
      P s
    , const v8::ScriptOrigin& origin
  ) {
    return v8::Script::New(s, const_cast<v8::ScriptOrigin *>(&origin));
  }

  template<>
  NAN_INLINE v8::Local<NanUnboundScript> NanNew<NanUnboundScript>(
      v8::Local<v8::String> s
  ) {
    return v8::Script::New(s);
  }

  NAN_INLINE v8::Local<v8::String> NanNew(
      v8::String::ExternalStringResource *resource) {
    return v8::String::NewExternal(resource);
  }

  NAN_INLINE v8::Local<v8::String> NanNew(
      v8::String::ExternalAsciiStringResource *resource) {
    return v8::String::NewExternal(resource);
  }

  template<>
  NAN_INLINE v8::Local<v8::BooleanObject> NanNew(bool value) {
    return v8::BooleanObject::New(value).As<v8::BooleanObject>();
  }

  template<>
  NAN_INLINE v8::Local<v8::StringObject>
  NanNew<v8::StringObject, v8::Local<v8::String> >(
      v8::Local<v8::String> value) {
    return v8::StringObject::New(value).As<v8::StringObject>();
  }

  template<>
  NAN_INLINE v8::Local<v8::StringObject>
  NanNew<v8::StringObject, v8::Handle<v8::String> >(
      v8::Handle<v8::String> value) {
    return v8::StringObject::New(value).As<v8::StringObject>();
  }

  template<>
  NAN_INLINE v8::Local<v8::NumberObject> NanNew<v8::NumberObject>(double val) {
    return v8::NumberObject::New(val).As<v8::NumberObject>();
  }

  template<>
  NAN_INLINE v8::Local<v8::Uint32> NanNew<v8::Uint32, int32_t>(int32_t val) {
    return v8::Uint32::NewFromUnsigned(val)->ToUint32();
  }

  template<>
  NAN_INLINE v8::Local<v8::Uint32> NanNew<v8::Uint32, uint32_t>(uint32_t val) {
    return v8::Uint32::NewFromUnsigned(val)->ToUint32();
  }

  template<>
  NAN_INLINE v8::Local<v8::Int32> NanNew<v8::Int32, int32_t>(int32_t val) {
    return v8::Int32::New(val)->ToInt32();
  }

  template<>
  NAN_INLINE v8::Local<v8::Int32> NanNew<v8::Int32, uint32_t>(uint32_t val) {
    return v8::Int32::New(val)->ToInt32();
  }

  template<>
  NAN_INLINE v8::Local<v8::String> NanNew<v8::String, uint8_t *>(
      uint8_t *arg
    , int length) {
    uint16_t *warg = new uint16_t[length];
    for (int i = 0; i < length; i++) {
      warg[i] = arg[i];
    }
    v8::Local<v8::String> retval = v8::String::New(warg, length);
    delete[] warg;
    return retval;
  }

  template<>
  NAN_INLINE v8::Local<v8::String> NanNew<v8::String, const uint8_t *>(
      const uint8_t *arg
    , int length) {
    uint16_t *warg = new uint16_t[length];
    for (int i = 0; i < length; i++) {
      warg[i] = arg[i];
    }
    v8::Local<v8::String> retval = v8::String::New(warg, length);
    delete[] warg;
    return retval;
  }

  template<>
  NAN_INLINE v8::Local<v8::String> NanNew<v8::String, uint8_t *>(uint8_t *arg) {
    int length = strlen(reinterpret_cast<char *>(arg));
    uint16_t *warg = new uint16_t[length];
    for (int i = 0; i < length; i++) {
      warg[i] = arg[i];
    }

    v8::Local<v8::String> retval = v8::String::New(warg, length);
    delete[] warg;
    return retval;
  }

  template<>
  NAN_INLINE v8::Local<v8::String> NanNew<v8::String, const uint8_t *>(
      const uint8_t *arg) {
    int length = strlen(reinterpret_cast<const char *>(arg));
    uint16_t *warg = new uint16_t[length];
    for (int i = 0; i < length; i++) {
      warg[i] = arg[i];
    }
    v8::Local<v8::String> retval = v8::String::New(warg, length);
    delete[] warg;
    return retval;
  }

  template<>
  NAN_INLINE v8::Local<v8::String> NanNew<v8::String>() {
    return v8::String::Empty();
  }

  NAN_INLINE void NanAddGCEpilogueCallback(
    v8::GCEpilogueCallback callback
  , v8::GCType gc_type_filter = v8::kGCTypeAll) {
    v8::V8::AddGCEpilogueCallback(callback, gc_type_filter);
  }
  NAN_INLINE void NanRemoveGCEpilogueCallback(
    v8::GCEpilogueCallback callback) {
    v8::V8::RemoveGCEpilogueCallback(callback);
  }
  NAN_INLINE void NanAddGCPrologueCallback(
    v8::GCPrologueCallback callback
  , v8::GCType gc_type_filter = v8::kGCTypeAll) {
    v8::V8::AddGCPrologueCallback(callback, gc_type_filter);
  }
  NAN_INLINE void NanRemoveGCPrologueCallback(
    v8::GCPrologueCallback callback) {
    v8::V8::RemoveGCPrologueCallback(callback);
  }
  NAN_INLINE void NanGetHeapStatistics(
    v8::HeapStatistics *heap_statistics) {
    v8::V8::GetHeapStatistics(heap_statistics);
  }

  template<typename T>
  NAN_INLINE void NanAssignPersistent(
      v8::Persistent<T>& handle
    , v8::Handle<T> obj) {
      handle.Dispose();
      handle = v8::Persistent<T>::New(obj);
  }

  template<typename T, typename P>
  struct _NanWeakCallbackInfo {
    typedef void (*Callback)(v8::Persistent<v8::Value> object, void* parameter);
    _NanWeakCallbackInfo(v8::Handle<T> handle, P* param, Callback cb) :
        parameter(param)
      , callback(cb)
      , persistent(v8::Persistent<T>::New(handle)) { }

    ~_NanWeakCallbackInfo() {
      persistent.Dispose();
      persistent.Clear();
    }

    P* const parameter;
    Callback const callback;
    v8::Persistent<T> persistent;
  };

  template<typename T, typename P>
  class _NanWeakCallbackData {
   public:
    _NanWeakCallbackData(_NanWeakCallbackInfo<T, P> *info)
      : info_(info) { }

    NAN_INLINE v8::Local<T> GetValue() const {
      return NanNew(info_->persistent);
    }
    NAN_INLINE P* GetParameter() const { return info_->parameter; }
    NAN_INLINE void Revive() const {
      info_->persistent.MakeWeak(info_, info_->callback);
    }
    NAN_INLINE void Dispose() const {
      delete info_;
    }

   private:
    _NanWeakCallbackInfo<T, P>* info_;
  };

# define NanGetInternalFieldPointer(object, index)                             \
    object->GetPointerFromInternalField(index)
# define NanSetInternalFieldPointer(object, index, value)                      \
    object->SetPointerInInternalField(index, value)

// do not use for declaration
# define NAN_WEAK_CALLBACK(name)                                               \
    template<typename T, typename P>                                           \
    static void name(                                                          \
      v8::Persistent<v8::Value> object, void *data) {                          \
        _NanWeakCallbackData<T, P> wcbd(                                       \
           static_cast<_NanWeakCallbackInfo<T, P>*>(data));                    \
        _Nan_Weak_Callback_ ## name(wcbd);                                     \
    }                                                                          \
                                                                               \
    template<typename T, typename P>                                           \
    NAN_INLINE void _Nan_Weak_Callback_ ## name(                               \
        const _NanWeakCallbackData<T, P> &data)

  template<typename T, typename P>
  NAN_INLINE void NanMakeWeakPersistent(
    v8::Handle<T> handle
  , P* parameter
  , typename _NanWeakCallbackInfo<T, P>::Callback callback) {
      _NanWeakCallbackInfo<T, P> *cbinfo =
        new _NanWeakCallbackInfo<T, P>(handle, parameter, callback);
      cbinfo->persistent.MakeWeak(cbinfo, callback);
  }

# define NanScope() v8::HandleScope scope
# define NanEscapableScope() v8::HandleScope scope
# define NanEscapeScope(val) scope.Close(val)
# define NanLocker() v8::Locker locker
# define NanUnlocker() v8::Unlocker unlocker
# define NanReturnValue(value) return scope.Close(value)
# define NanReturnUndefined() return v8::Undefined()
# define NanReturnNull() return v8::Null()
# define NanReturnEmptyString() return v8::String::Empty()
# define NanObjectWrapHandle(obj) v8::Local<v8::Object>::New(obj->handle_)

# define _NAN_ERROR(fun, errmsg)                                               \
    fun(v8::String::New(errmsg))

# define _NAN_THROW_ERROR(fun, errmsg)                                         \
    do {                                                                       \
      NanScope();                                                              \
      return v8::Local<v8::Value>::New(                                        \
        v8::ThrowException(_NAN_ERROR(fun, errmsg)));                          \
    } while (0);

  NAN_INLINE v8::Local<v8::Value> NanError(const char* errmsg) {
    return _NAN_ERROR(v8::Exception::Error, errmsg);
  }

  NAN_INLINE v8::Local<v8::Value> NanThrowError(const char* errmsg) {
    _NAN_THROW_ERROR(v8::Exception::Error, errmsg);
  }

  NAN_INLINE v8::Local<v8::Value> NanThrowError(
      v8::Handle<v8::Value> error
  ) {
    NanScope();
    return v8::Local<v8::Value>::New(v8::ThrowException(error));
  }

  NAN_INLINE v8::Local<v8::Value> NanError(
      const char *msg
    , const int errorNumber
  ) {
    v8::Local<v8::Value> err = v8::Exception::Error(v8::String::New(msg));
    v8::Local<v8::Object> obj = err.As<v8::Object>();
    obj->Set(v8::String::New("code"), v8::Int32::New(errorNumber));
    return err;
  }

  NAN_INLINE v8::Local<v8::Value> NanThrowError(
      const char *msg
    , const int errorNumber
  ) {
    return NanThrowError(NanError(msg, errorNumber));
  }

  NAN_INLINE v8::Local<v8::Value> NanTypeError(const char* errmsg) {
    return _NAN_ERROR(v8::Exception::TypeError, errmsg);
  }

  NAN_INLINE v8::Local<v8::Value> NanThrowTypeError(
      const char* errmsg
  ) {
    _NAN_THROW_ERROR(v8::Exception::TypeError, errmsg);
  }

  NAN_INLINE v8::Local<v8::Value> NanRangeError(
      const char* errmsg
  ) {
    return _NAN_ERROR(v8::Exception::RangeError, errmsg);
  }

  NAN_INLINE v8::Local<v8::Value> NanThrowRangeError(
      const char* errmsg
  ) {
    _NAN_THROW_ERROR(v8::Exception::RangeError, errmsg);
  }

  template<typename T>
  NAN_INLINE void NanDisposePersistent(
      v8::Persistent<T> &handle) {  // NOLINT(runtime/references)
    handle.Dispose();
    handle.Clear();
  }

  NAN_INLINE v8::Local<v8::Object> NanNewBufferHandle (
      char *data
    , size_t length
    , node::Buffer::free_callback callback
    , void *hint
  ) {
    return NanNew<v8::Object>(
        node::Buffer::New(data, length, callback, hint)->handle_);
  }

  NAN_INLINE v8::Local<v8::Object> NanNewBufferHandle (
      const char *data
    , uint32_t size
  ) {
#if NODE_MODULE_VERSION >= 0x000B
    return NanNew<v8::Object>(node::Buffer::New(data, size)->handle_);
#else
    return NanNew<v8::Object>(
      node::Buffer::New(const_cast<char*>(data), size)->handle_);
#endif
  }

  NAN_INLINE v8::Local<v8::Object> NanNewBufferHandle (uint32_t size) {
    return NanNew<v8::Object>(node::Buffer::New(size)->handle_);
  }

  NAN_INLINE void FreeData(char *data, void *hint) {
    delete[] data;
  }

  NAN_INLINE v8::Local<v8::Object> NanBufferUse(
      char* data
    , uint32_t size
  ) {
    return NanNew<v8::Object>(
        node::Buffer::New(data, size, FreeData, NULL)->handle_);
  }

  NAN_INLINE bool NanHasInstance(
      v8::Persistent<v8::FunctionTemplate>& function_template
    , v8::Handle<v8::Value> value
  ) {
    return function_template->HasInstance(value);
  }

  NAN_INLINE v8::Local<v8::Context> NanNewContextHandle(
      v8::ExtensionConfiguration* extensions = NULL
    , v8::Handle<v8::ObjectTemplate> tmpl = v8::Handle<v8::ObjectTemplate>()
    , v8::Handle<v8::Value> obj = v8::Handle<v8::Value>()
  ) {
    v8::Persistent<v8::Context> ctx = v8::Context::New(extensions, tmpl, obj);
    v8::Local<v8::Context> lctx = NanNew<v8::Context>(ctx);
    ctx.Dispose();
    return lctx;
  }

  NAN_INLINE v8::Local<NanBoundScript> NanCompileScript(
      v8::Local<v8::String> s
    , const v8::ScriptOrigin& origin
  ) {
    return v8::Script::Compile(s, const_cast<v8::ScriptOrigin *>(&origin));
  }

  NAN_INLINE v8::Local<NanBoundScript> NanCompileScript(
    v8::Local<v8::String> s
  ) {
    return v8::Script::Compile(s);
  }

  NAN_INLINE v8::Local<v8::Value> NanRunScript(v8::Local<v8::Script> script) {
    return script->Run();
  }

#endif  // NODE_MODULE_VERSION

typedef void (*NanFreeCallback)(char *data, void *hint);

#define NAN_METHOD(name) _NAN_METHOD_RETURN_TYPE name(_NAN_METHOD_ARGS)
#define NAN_GETTER(name)                                                       \
    _NAN_GETTER_RETURN_TYPE name(                                              \
        v8::Local<v8::String> property                                         \
      , _NAN_GETTER_ARGS)
#define NAN_SETTER(name)                                                       \
    _NAN_SETTER_RETURN_TYPE name(                                              \
        v8::Local<v8::String> property                                         \
      , v8::Local<v8::Value> value                                             \
      , _NAN_SETTER_ARGS)
#define NAN_PROPERTY_GETTER(name)                                              \
    _NAN_PROPERTY_GETTER_RETURN_TYPE name(                                     \
        v8::Local<v8::String> property                                         \
      , _NAN_PROPERTY_GETTER_ARGS)
#define NAN_PROPERTY_SETTER(name)                                              \
    _NAN_PROPERTY_SETTER_RETURN_TYPE name(                                     \
        v8::Local<v8::String> property                                         \
      , v8::Local<v8::Value> value                                             \
      , _NAN_PROPERTY_SETTER_ARGS)
#define NAN_PROPERTY_ENUMERATOR(name)                                          \
    _NAN_PROPERTY_ENUMERATOR_RETURN_TYPE name(_NAN_PROPERTY_ENUMERATOR_ARGS)
#define NAN_PROPERTY_DELETER(name)                                             \
    _NAN_PROPERTY_DELETER_RETURN_TYPE name(                                    \
        v8::Local<v8::String> property                                         \
      , _NAN_PROPERTY_DELETER_ARGS)
#define NAN_PROPERTY_QUERY(name)                                               \
    _NAN_PROPERTY_QUERY_RETURN_TYPE name(                                      \
        v8::Local<v8::String> property                                         \
      , _NAN_PROPERTY_QUERY_ARGS)
# define NAN_INDEX_GETTER(name)                                                \
    _NAN_INDEX_GETTER_RETURN_TYPE name(uint32_t index, _NAN_INDEX_GETTER_ARGS)
#define NAN_INDEX_SETTER(name)                                                 \
    _NAN_INDEX_SETTER_RETURN_TYPE name(                                        \
        uint32_t index                                                         \
      , v8::Local<v8::Value> value                                             \
      , _NAN_INDEX_SETTER_ARGS)
#define NAN_INDEX_ENUMERATOR(name)                                             \
    _NAN_INDEX_ENUMERATOR_RETURN_TYPE name(_NAN_INDEX_ENUMERATOR_ARGS)
#define NAN_INDEX_DELETER(name)                                                \
    _NAN_INDEX_DELETER_RETURN_TYPE name(                                       \
        uint32_t index                                                         \
      , _NAN_INDEX_DELETER_ARGS)
#define NAN_INDEX_QUERY(name)                                                  \
    _NAN_INDEX_QUERY_RETURN_TYPE name(uint32_t index, _NAN_INDEX_QUERY_ARGS)

class NanCallback {
 public:
  NanCallback() {
    NanScope();
    v8::Local<v8::Object> obj = NanNew<v8::Object>();
    NanAssignPersistent(handle, obj);
  }

  explicit NanCallback(const v8::Handle<v8::Function> &fn) {
    NanScope();
    v8::Local<v8::Object> obj = NanNew<v8::Object>();
    NanAssignPersistent(handle, obj);
    SetFunction(fn);
  }

  ~NanCallback() {
    if (handle.IsEmpty()) return;
    NanDisposePersistent(handle);
  }

  NAN_INLINE void SetFunction(const v8::Handle<v8::Function> &fn) {
    NanScope();
    NanNew(handle)->Set(NanSymbol("callback"), fn);
  }

  NAN_INLINE v8::Local<v8::Function> GetFunction () {
    return NanNew(handle)->Get(NanSymbol("callback"))
        .As<v8::Function>();
  }

  void Call(int argc, v8::Handle<v8::Value> argv[]) {
    NanScope();
#if (NODE_MODULE_VERSION > 0x000B)  // 0.11.12+
    v8::Local<v8::Function> callback = NanNew(handle)->
        Get(NanSymbol("callback")).As<v8::Function>();
    node::MakeCallback(
        nan_isolate
      , nan_isolate->GetCurrentContext()->Global()
      , callback
      , argc
      , argv
    );
#else
#if NODE_VERSION_AT_LEAST(0, 8, 0)
    v8::Local<v8::Function> callback = NanNew(handle)->
        Get(NanSymbol("callback")).As<v8::Function>();
    node::MakeCallback(
        v8::Context::GetCurrent()->Global()
      , callback
      , argc
      , argv
    );
#else
    node::MakeCallback(handle, "callback", argc, argv);
#endif
#endif
  }

 private:
  v8::Persistent<v8::Object> handle;
};

/* abstract */ class NanAsyncWorker {
 public:
  explicit NanAsyncWorker(NanCallback *callback) : callback(callback) {
    request.data = this;
    errmsg = NULL;

    NanScope();
    v8::Local<v8::Object> obj = NanNew<v8::Object>();
    NanAssignPersistent(persistentHandle, obj);
  }

  virtual ~NanAsyncWorker() {
    NanScope();

    if (!persistentHandle.IsEmpty())
      NanDisposePersistent(persistentHandle);
    if (callback)
      delete callback;
    if (errmsg)
      delete errmsg;
  }

  virtual void WorkComplete() {
    NanScope();

    if (errmsg == NULL)
      HandleOKCallback();
    else
      HandleErrorCallback();
    delete callback;
    callback = NULL;
  }

  NAN_INLINE void SaveToPersistent(const char *key, v8::Local<v8::Object> &obj) {
    v8::Local<v8::Object> handle = NanNew(persistentHandle);
    handle->Set(NanSymbol(key), obj);
  }

  v8::Local<v8::Object> GetFromPersistent(const char *key) {
    NanEscapableScope();
    v8::Local<v8::Object> handle = NanNew(persistentHandle);
    return NanEscapeScope(handle->Get(NanSymbol(key)).As<v8::Object>());
  }

  virtual void Execute() = 0;

  uv_work_t request;

 protected:
  v8::Persistent<v8::Object> persistentHandle;
  NanCallback *callback;
  const char *errmsg;

  virtual void HandleOKCallback() {
    NanScope();

    callback->Call(0, NULL);
  }

  virtual void HandleErrorCallback() {
    NanScope();

    v8::Local<v8::Value> argv[] = {
        v8::Exception::Error(NanNew<v8::String>(errmsg))
    };
    callback->Call(1, argv);
  }
};

NAN_INLINE void NanAsyncExecute (uv_work_t* req) {
  NanAsyncWorker *worker = static_cast<NanAsyncWorker*>(req->data);
  worker->Execute();
}

NAN_INLINE void NanAsyncExecuteComplete (uv_work_t* req) {
  NanAsyncWorker* worker = static_cast<NanAsyncWorker*>(req->data);
  worker->WorkComplete();
  delete worker;
}

NAN_INLINE void NanAsyncQueueWorker (NanAsyncWorker* worker) {
  uv_queue_work(
      uv_default_loop()
    , &worker->request
    , NanAsyncExecute
    , (uv_after_work_cb)NanAsyncExecuteComplete
  );
}

//// Base 64 ////

#define _nan_base64_encoded_size(size) ((size + 2 - ((size + 2) % 3)) / 3 * 4)

// Doesn't check for padding at the end.  Can be 1-2 bytes over.
NAN_INLINE size_t _nan_base64_decoded_size_fast(size_t size) {
  size_t remainder = size % 4;

  size = (size / 4) * 3;
  if (remainder) {
    if (size == 0 && remainder == 1) {
      // special case: 1-byte input cannot be decoded
      size = 0;
    } else {
      // non-padded input, add 1 or 2 extra bytes
      size += 1 + (remainder == 3);
    }
  }

  return size;
}

template<typename T>
NAN_INLINE size_t _nan_base64_decoded_size(
    const T* src
  , size_t size
) {
  if (size == 0)
    return 0;

  if (src[size - 1] == '=')
    size--;
  if (size > 0 && src[size - 1] == '=')
    size--;

  return _nan_base64_decoded_size_fast(size);
}

// supports regular and URL-safe base64
static const int _nan_unbase64_table[] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -1, -1, -2, -1, -1
  , -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
  , -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 62, -1, 62, -1, 63
  , 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, -1, -1, -1, -1, -1, -1
  , -1,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14
  , 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, -1, -1, -1, -1, 63
  , -1, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40
  , 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, -1, -1, -1, -1, -1
  , -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
  , -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
  , -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
  , -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
  , -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
  , -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
  , -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
  , -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
};

#define _nan_unbase64(x) _nan_unbase64_table[(uint8_t)(x)]

template<typename T> static size_t _nan_base64_decode(
    char* buf
  , size_t len
  , const T* src
  , const size_t srcLen
) {
  char* dst = buf;
  char* dstEnd = buf + len;
  const T* srcEnd = src + srcLen;

  while (src < srcEnd && dst < dstEnd) {
    ptrdiff_t remaining = srcEnd - src;
    char a, b, c, d;

    while (_nan_unbase64(*src) < 0 && src < srcEnd) src++, remaining--;
    if (remaining == 0 || *src == '=') break;
    a = _nan_unbase64(*src++);

    while (_nan_unbase64(*src) < 0 && src < srcEnd) src++, remaining--;
    if (remaining <= 1 || *src == '=') break;
    b = _nan_unbase64(*src++);

    *dst++ = (a << 2) | ((b & 0x30) >> 4);
    if (dst == dstEnd) break;

    while (_nan_unbase64(*src) < 0 && src < srcEnd) src++, remaining--;
    if (remaining <= 2 || *src == '=') break;
    c = _nan_unbase64(*src++);

    *dst++ = ((b & 0x0F) << 4) | ((c & 0x3C) >> 2);
    if (dst == dstEnd) break;

    while (_nan_unbase64(*src) < 0 && src < srcEnd) src++, remaining--;
    if (remaining <= 3 || *src == '=') break;
    d = _nan_unbase64(*src++);

    *dst++ = ((c & 0x03) << 6) | (d & 0x3F);
  }

  return dst - buf;
}

//// HEX ////

template<typename T> unsigned _nan_hex2bin(T c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
  if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
  return static_cast<unsigned>(-1);
}

template<typename T> static size_t _nan_hex_decode(
    char* buf
  , size_t len
  , const T* src
  , const size_t srcLen
) {
  size_t i;
  for (i = 0; i < len && i * 2 + 1 < srcLen; ++i) {
    unsigned a = _nan_hex2bin(src[i * 2 + 0]);
    unsigned b = _nan_hex2bin(src[i * 2 + 1]);
    if (!~a || !~b) return i;
    buf[i] = a * 16 + b;
  }

  return i;
}

static bool _NanGetExternalParts(
    v8::Handle<v8::Value> val
  , const char** data
  , size_t* len
) {
  if (node::Buffer::HasInstance(val)) {
    *data = node::Buffer::Data(val.As<v8::Object>());
    *len = node::Buffer::Length(val.As<v8::Object>());
    return true;
  }

  assert(val->IsString());
  v8::Local<v8::String> str = NanNew<v8::String>(val.As<v8::String>());

  if (str->IsExternalAscii()) {
    const v8::String::ExternalAsciiStringResource* ext;
    ext = str->GetExternalAsciiStringResource();
    *data = ext->data();
    *len = ext->length();
    return true;

  } else if (str->IsExternal()) {
    const v8::String::ExternalStringResource* ext;
    ext = str->GetExternalStringResource();
    *data = reinterpret_cast<const char*>(ext->data());
    *len = ext->length();
    return true;
  }

  return false;
}

namespace Nan {
  enum Encoding {ASCII, UTF8, BASE64, UCS2, BINARY, HEX, BUFFER};
}

NAN_INLINE void* NanRawString(
    v8::Handle<v8::Value> from
  , enum Nan::Encoding encoding
  , size_t *datalen
  , void *buf
  , size_t buflen
  , int flags
) {
  NanScope();

  size_t sz_;
  size_t term_len = !(flags & v8::String::NO_NULL_TERMINATION);
  char *data = NULL;
  size_t len;
  bool is_extern = _NanGetExternalParts(
      from
    , const_cast<const char**>(&data)
    , &len);

  if (is_extern && !term_len) {
    NanSetPointerSafe(datalen, len);
    return data;
  }

  v8::Local<v8::String> toStr = from->ToString();

  char *to = static_cast<char *>(buf);

  switch (encoding) {
    case Nan::ASCII:
#if NODE_MODULE_VERSION < 0x000C
      sz_ = toStr->Length();
      if (to == NULL) {
        to = new char[sz_ + term_len];
      } else {
        assert(buflen >= sz_ + term_len && "too small buffer");
      }
      NanSetPointerSafe<size_t>(
          datalen
        , toStr->WriteAscii(to, 0, static_cast<int>(sz_ + term_len), flags));
      return to;
#endif
    case Nan::BINARY:
    case Nan::BUFFER:
      sz_ = toStr->Length();
      if (to == NULL) {
        to = new char[sz_ + term_len];
      } else {
        assert(buflen >= sz_ + term_len && "too small buffer");
      }
#if NODE_MODULE_VERSION < 0x000C
      {
        uint16_t* twobytebuf = new uint16_t[sz_ + term_len];

        size_t len = toStr->Write(twobytebuf, 0,
          static_cast<int>(sz_ + term_len), flags);

        for (size_t i = 0; i < sz_ + term_len && i < len + term_len; i++) {
          unsigned char *b = reinterpret_cast<unsigned char*>(&twobytebuf[i]);
          to[i] = *b;
        }

        NanSetPointerSafe<size_t>(datalen, len);

        delete[] twobytebuf;
        return to;
      }
#else
      NanSetPointerSafe<size_t>(
        datalen,
        toStr->WriteOneByte(
            reinterpret_cast<uint8_t *>(to)
          , 0
          , static_cast<int>(sz_ + term_len)
          , flags));
      return to;
#endif
    case Nan::UTF8:
      sz_ = toStr->Utf8Length();
      if (to == NULL) {
        to = new char[sz_ + term_len];
      } else {
        assert(buflen >= sz_ + term_len && "too small buffer");
      }
      NanSetPointerSafe<size_t>(
          datalen
        , toStr->WriteUtf8(to, static_cast<int>(sz_ + term_len)
            , NULL, flags)
          - term_len);
      return to;
    case Nan::BASE64:
      {
        v8::String::Value value(toStr);
        sz_ = _nan_base64_decoded_size(*value, value.length());
        if (to == NULL) {
          to = new char[sz_ + term_len];
        } else {
          assert(buflen >= sz_ + term_len);
        }
        NanSetPointerSafe<size_t>(
            datalen
          , _nan_base64_decode(to, sz_, *value, value.length()));
        if (term_len) {
          to[sz_] = '\0';
        }
        return to;
      }
    case Nan::UCS2:
      {
        sz_ = toStr->Length();
        if (to == NULL) {
          to = new char[(sz_ + term_len) * 2];
        } else {
          assert(buflen >= (sz_ + term_len) * 2 && "too small buffer");
        }

        int bc = 2 * toStr->Write(
            reinterpret_cast<uint16_t *>(to)
          , 0
          , static_cast<int>(sz_ + term_len)
          , flags);
        NanSetPointerSafe<size_t>(datalen, bc);
        return to;
      }
    case Nan::HEX:
      {
        v8::String::Value value(toStr);
        sz_ = value.length();
        assert(!(sz_ & 1) && "bad hex data");
        if (to == NULL) {
          to = new char[sz_ / 2 + term_len];
        } else {
          assert(buflen >= sz_ / 2 + term_len && "too small buffer");
        }
        NanSetPointerSafe<size_t>(
            datalen
          , _nan_hex_decode(to, sz_ / 2, *value, value.length()));
      }
      if (term_len) {
        to[sz_ / 2] = '\0';
      }
      return to;
    default:
      assert(0 && "unknown encoding");
  }
  return to;
}

NAN_INLINE char* NanCString(
    v8::Handle<v8::Value> from
  , size_t *datalen
  , char *buf = NULL
  , size_t buflen = 0
  , int flags = v8::String::NO_OPTIONS
) {
    return static_cast<char *>(
      NanRawString(from, Nan::UTF8, datalen, buf, buflen, flags)
    );
}

#endif  // NAN_H_
