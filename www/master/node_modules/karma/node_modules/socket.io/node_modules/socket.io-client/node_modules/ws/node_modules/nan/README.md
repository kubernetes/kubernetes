Native Abstractions for Node.js
===============================

**A header file filled with macro and utility goodness for making add-on development for Node.js easier across versions 0.8, 0.10 and 0.11, and eventually 0.12.**

***Current version: 1.0.0*** *(See [nan.h](https://github.com/rvagg/nan/blob/master/nan.h) for complete ChangeLog)*

[![NPM](https://nodei.co/npm/nan.png?downloads=true)](https://nodei.co/npm/nan/) [![NPM](https://nodei.co/npm-dl/nan.png?months=6)](https://nodei.co/npm/nan/)

Thanks to the crazy changes in V8 (and some in Node core), keeping native addons compiling happily across versions, particularly 0.10 to 0.11/0.12, is a minor nightmare. The goal of this project is to store all logic necessary to develop native Node.js addons without having to inspect `NODE_MODULE_VERSION` and get yourself into a macro-tangle.

This project also contains some helper utilities that make addon development a bit more pleasant.

 * **[News & Updates](#news)**
 * **[Usage](#usage)**
 * **[Example](#example)**
 * **[API](#api)**

<a name="news"></a>
## News & Updates

### May-2013: Major changes for V8 3.25 / Node 0.11.13

Node 0.11.11 and 0.11.12 were both broken releases for native add-ons, you simply can't properly compile against either of them for different reasons. But we now have a 0.11.13 release that jumps a couple of versions of V8 ahead and includes some more, major (traumatic) API changes.

Because we are now nearing Node 0.12 and estimate that the version of V8 we are using in Node 0.11.13 will be close to the API we get for 0.12, we have taken the opportunity to not only *fix* NAN for 0.11.13 but make some major changes to improve the NAN API.

We have **removed support for Node 0.11 versions prior to 0.11.13**, (although our tests are still passing for 0.11.10). As usual, our tests are run against (and pass) the last 5 versions of Node 0.8 and Node 0.10. We also include Node 0.11.13 obviously.

The major change is something that [Benjamin Byholm](kkoopa) has put many hours in to. We now have a fantastic new `NanNew<T>(args)` interface for creating new `Local`s, this replaces `NanNewLocal()` and much more. If you look in [./nan.h](nan.h) you'll see a large number of overloaded versions of this method. In general you should be able to `NanNew<Type>(arguments)` for any type you want to make a `Local` from. This includes `Persistent` types, so we now have a `Local<T> NanNew(const Persistent<T> arg)` to replace `NanPersistentToLocal()`.

We also now have `NanUndefined()`, `NanNull()`, `NanTrue()` and `NanFalse()`. Mainly because of the new requirement for an `Isolate` argument for each of the native V8 versions of this.

V8 has now introduced an `EscapableHandleScope` from which you `scope.Escape(Local<T> value)` to *return* a value from a one scope to another. This replaces the standard `HandleScope` and `scope.Close(Local<T> value)`, although `HandleScope` still exists for when you don't need to return a handle to the caller. For NAN we are exposing it as `NanEscapableScope()` and `NanEscapeScope()`, while `NanScope()` is still how you create a new scope that doesn't need to return handles. For older versions of Node/V8, it'll still map to the older `HandleScope` functionality.

`NanFromV8String()` was deprecated and has now been removed. You should use `NanCString()` or `NanRawString()` instead.

Because `node::MakeCallback()` now takes an `Isolate`, and because it doesn't exist in older versions of Node, we've introduced `NanMakeCallabck()`. You should *always* use this when calling a JavaScript function from C++.

There's lots more, check out the Changelog in nan.h or look through [#86](https://github.com/rvagg/nan/pull/86) for all the gory details.

### Dec-2013: NanCString and NanRawString

Two new functions have been introduced to replace the functionality that's been provided by `NanFromV8String` until now. NanCString has sensible defaults so it's super easy to fetch a null-terminated c-style string out of a `v8::String`. `NanFromV8String` is still around and has defaults that allow you to pass a single handle to fetch a `char*` while `NanRawString` requires a little more attention to arguments.

### Nov-2013: Node 0.11.9+ breaking V8 change

The version of V8 that's shipping with Node 0.11.9+ has changed the signature for new `Local`s to: `v8::Local<T>::New(isolate, value)`, i.e. introducing the `isolate` argument and therefore breaking all new `Local` declarations for previous versions. NAN 0.6+ now includes a `NanNewLocal<T>(value)` that can be used in place to work around this incompatibility and maintain compatibility with 0.8->0.11.9+ (minus a few early 0.11 releases).

For example, if you wanted to return a `null` on a callback you will have to change the argument from `v8::Local<v8::Value>::New(v8::Null())` to `NanNewLocal<v8::Value>(v8::Null())`.

### Nov-2013: Change to binding.gyp `"include_dirs"` for NAN

Inclusion of NAN in a project's binding.gyp is now greatly simplified. You can now just use `"<!(node -e \"require('nan')\")"` in your `"include_dirs"`, see example below (note Windows needs the quoting around `require` to be just right: `"require('nan')"` with appropriate `\` escaping).

<a name="usage"></a>
## Usage

Simply add **NAN** as a dependency in the *package.json* of your Node addon:

``` bash
$ npm install --save nan
```

Pull in the path to **NAN** in your *binding.gyp* so that you can use `#include <nan.h>` in your *.cpp* files:

``` python
"include_dirs" : [
    "<!(node -e \"require('nan')\")"
]
```

This works like a `-I<path-to-NAN>` when compiling your addon.

<a name="example"></a>
## Example

See **[LevelDOWN](https://github.com/rvagg/node-leveldown/pull/48)** for a full example of **NAN** in use.

For a simpler example, see the **[async pi estimation example](https://github.com/rvagg/nan/tree/master/examples/async_pi_estimate)** in the examples directory for full code and an explanation of what this Monte Carlo Pi estimation example does. Below are just some parts of the full example that illustrate the use of **NAN**.

Compare to the current 0.10 version of this example, found in the [node-addon-examples](https://github.com/rvagg/node-addon-examples/tree/master/9_async_work) repository and also a 0.11 version of the same found [here](https://github.com/kkoopa/node-addon-examples/tree/5c01f58fc993377a567812597e54a83af69686d7/9_async_work).

Note that there is no embedded version sniffing going on here and also the async work is made much simpler, see below for details on the `NanAsyncWorker` class.

```c++
// addon.cc
#include <node.h>
#include <nan.h>
// ...

using v8::FunctionTemplate;
using v8::Handle;
using v8::Object;

void InitAll(Handle<Object> exports) {
  exports->Set(NanSymbol("calculateSync"),
    NanNew<FunctionTemplate>(CalculateSync)->GetFunction());

  exports->Set(NanSymbol("calculateAsync"),
    NanNew<FunctionTemplate>(CalculateAsync)->GetFunction());
}

NODE_MODULE(addon, InitAll)
```

```c++
// sync.h
#include <node.h>
#include <nan.h>

NAN_METHOD(CalculateSync);
```

```c++
// sync.cc
#include <node.h>
#include <nan.h>
#include "./sync.h"
// ...

using v8::Number;

// Simple synchronous access to the `Estimate()` function
NAN_METHOD(CalculateSync) {
  NanScope();

  // expect a number as the first argument
  int points = args[0]->Uint32Value();
  double est = Estimate(points);

  NanReturnValue(NanNew<Number>(est));
}
```

```c++
// async.cc
#include <node.h>
#include <nan.h>
#include "./async.h"

// ...

using v8::Function;
using v8::Local;
using v8::Null;
using v8::Number;
using v8::Value;

class PiWorker : public NanAsyncWorker {
 public:
  PiWorker(NanCallback *callback, int points)
    : NanAsyncWorker(callback), points(points) {}
  ~PiWorker() {}

  // Executed inside the worker-thread.
  // It is not safe to access V8, or V8 data structures
  // here, so everything we need for input and output
  // should go on `this`.
  void Execute () {
    estimate = Estimate(points);
  }

  // Executed when the async work is complete
  // this function will be run inside the main event loop
  // so it is safe to use V8 again
  void HandleOKCallback () {
    NanScope();

    Local<Value> argv[] = {
        NanNew(NanNull())
      , NanNew<Number>(estimate)
    };

    callback->Call(2, argv);
  };

 private:
  int points;
  double estimate;
};

// Asynchronous access to the `Estimate()` function
NAN_METHOD(CalculateAsync) {
  NanScope();

  int points = args[0]->Uint32Value();
  NanCallback *callback = new NanCallback(args[1].As<Function>());

  NanAsyncQueueWorker(new PiWorker(callback, points));
  NanReturnUndefined();
}
```

<a name="api"></a>
## API

 * <a href="#api_nan_method"><b><code>NAN_METHOD</code></b></a>
 * <a href="#api_nan_getter"><b><code>NAN_GETTER</code></b></a>
 * <a href="#api_nan_setter"><b><code>NAN_SETTER</code></b></a>
 * <a href="#api_nan_property_getter"><b><code>NAN_PROPERTY_GETTER</code></b></a>
 * <a href="#api_nan_property_setter"><b><code>NAN_PROPERTY_SETTER</code></b></a>
 * <a href="#api_nan_property_enumerator"><b><code>NAN_PROPERTY_ENUMERATOR</code></b></a>
 * <a href="#api_nan_property_deleter"><b><code>NAN_PROPERTY_DELETER</code></b></a>
 * <a href="#api_nan_property_query"><b><code>NAN_PROPERTY_QUERY</code></b></a>
 * <a href="#api_nan_index_getter"><b><code>NAN_INDEX_GETTER</code></b></a>
 * <a href="#api_nan_index_setter"><b><code>NAN_INDEX_SETTER</code></b></a>
 * <a href="#api_nan_index_enumerator"><b><code>NAN_INDEX_ENUMERATOR</code></b></a>
 * <a href="#api_nan_index_deleter"><b><code>NAN_INDEX_DELETER</code></b></a>
 * <a href="#api_nan_index_query"><b><code>NAN_INDEX_QUERY</code></b></a>
 * <a href="#api_nan_weak_callback"><b><code>NAN_WEAK_CALLBACK</code></b></a>
 * <a href="#api_nan_deprecated"><b><code>NAN_DEPRECATED</code></b></a>
 * <a href="#api_nan_inline"><b><code>NAN_INLINE</code></b></a>
 * <a href="#api_nan_new"><b><code>NanNew</code></b></a>
 * <a href="#api_nan_undefined"><b><code>NanUndefined</code></b></a>
 * <a href="#api_nan_null"><b><code>NanNull</code></b></a>
 * <a href="#api_nan_true"><b><code>NanTrue</code></b></a>
 * <a href="#api_nan_false"><b><code>NanFalse</code></b></a>
 * <a href="#api_nan_return_value"><b><code>NanReturnValue</code></b></a>
 * <a href="#api_nan_return_undefined"><b><code>NanReturnUndefined</code></b></a>
 * <a href="#api_nan_return_null"><b><code>NanReturnNull</code></b></a>
 * <a href="#api_nan_return_empty_string"><b><code>NanReturnEmptyString</code></b></a>
 * <a href="#api_nan_scope"><b><code>NanScope</code></b></a>
 * <a href="#api_nan_escapable_scope"><b><code>NanEscapableScope</code></b></a>
 * <a href="#api_nan_escape_scope"><b><code>NanEscapeScope</code></b></a>
 * <a href="#api_nan_locker"><b><code>NanLocker</code></b></a>
 * <a href="#api_nan_unlocker"><b><code>NanUnlocker</code></b></a>
 * <a href="#api_nan_get_internal_field_pointer"><b><code>NanGetInternalFieldPointer</code></b></a>
 * <a href="#api_nan_set_internal_field_pointer"><b><code>NanSetInternalFieldPointer</code></b></a>
 * <a href="#api_nan_object_wrap_handle"><b><code>NanObjectWrapHandle</code></b></a>
 * <a href="#api_nan_symbol"><b><code>NanSymbol</code></b></a>
 * <a href="#api_nan_get_pointer_safe"><b><code>NanGetPointerSafe</code></b></a>
 * <a href="#api_nan_set_pointer_safe"><b><code>NanSetPointerSafe</code></b></a>
 * <a href="#api_nan_raw_string"><b><code>NanRawString</code></b></a>
 * <a href="#api_nan_c_string"><b><code>NanCString</code></b></a>
 * <a href="#api_nan_boolean_option_value"><b><code>NanBooleanOptionValue</code></b></a>
 * <a href="#api_nan_uint32_option_value"><b><code>NanUInt32OptionValue</code></b></a>
 * <a href="#api_nan_error"><b><code>NanError</code></b>, <b><code>NanTypeError</code></b>, <b><code>NanRangeError</code></b></a>
 * <a href="#api_nan_throw_error"><b><code>NanThrowError</code></b>, <b><code>NanThrowTypeError</code></b>, <b><code>NanThrowRangeError</code></b>, <b><code>NanThrowError(Handle<Value>)</code></b>, <b><code>NanThrowError(Handle<Value>, int)</code></b></a>
 * <a href="#api_nan_new_buffer_handle"><b><code>NanNewBufferHandle(char *, size_t, FreeCallback, void *)</code></b>, <b><code>NanNewBufferHandle(char *, uint32_t)</code></b>, <b><code>NanNewBufferHandle(uint32_t)</code></b></a>
 * <a href="#api_nan_buffer_use"><b><code>NanBufferUse(char *, uint32_t)</code></b></a>
 * <a href="#api_nan_new_context_handle"><b><code>NanNewContextHandle</code></b></a>
 * <a href="#api_nan_get_current_context"><b><code>NanGetCurrentContext</code></b></a>
 * <a href="#api_nan_has_instance"><b><code>NanHasInstance</code></b></a>
 * <a href="#api_nan_dispose_persistent"><b><code>NanDisposePersistent</code></b></a>
 * <a href="#api_nan_assign_persistent"><b><code>NanAssignPersistent</code></b></a>
 * <a href="#api_nan_make_weak_persistent"><b><code>NanMakeWeakPersistent</code></b></a>
 * <a href="#api_nan_set_template"><b><code>NanSetTemplate</code></b></a>
 * <a href="#api_nan_make_callback"><b><code>NanMakeCallback</code></b></a>
 * <a href="#api_nan_compile_script"><b><code>NanCompileScript</code></b></a>
 * <a href="#api_nan_run_script"><b><code>NanRunScript</code></b></a>
 * <a href="#api_nan_adjust_external_memory"><b><code>NanAdjustExternalMemory</code></b></a>
 * <a href="#api_nan_add_gc_epilogue_callback"><b><code>NanAddGCEpilogueCallback</code></b></a>
 * <a href="#api_nan_add_gc_prologue_callback"><b><code>NanAddGCPrologueCallback</code></b></a>
 * <a href="#api_nan_remove_gc_epilogue_callback"><b><code>NanRemoveGCEpilogueCallback</code></b></a>
 * <a href="#api_nan_remove_gc_prologue_callback"><b><code>NanRemoveGCPrologueCallback</code></b></a>
 * <a href="#api_nan_get_heap_statistics"><b><code>NanGetHeapStatistics</code></b></a>
 * <a href="#api_nan_callback"><b><code>NanCallback</code></b></a>
 * <a href="#api_nan_async_worker"><b><code>NanAsyncWorker</code></b></a>
 * <a href="#api_nan_async_queue_worker"><b><code>NanAsyncQueueWorker</code></b></a>

<a name="api_nan_method"></a>
### NAN_METHOD(methodname)

Use `NAN_METHOD` to define your V8 accessible methods:

```c++
// .h:
class Foo : public node::ObjectWrap {
  ...

  static NAN_METHOD(Bar);
  static NAN_METHOD(Baz);
}


// .cc:
NAN_METHOD(Foo::Bar) {
  ...
}

NAN_METHOD(Foo::Baz) {
  ...
}
```

The reason for this macro is because of the method signature change in 0.11:

```c++
// 0.10 and below:
Handle<Value> name(const Arguments& args)

// 0.11 and above
void name(const FunctionCallbackInfo<Value>& args)
```

The introduction of `FunctionCallbackInfo` brings additional complications:

<a name="api_nan_getter"></a>
### NAN_GETTER(methodname)

Use `NAN_GETTER` to declare your V8 accessible getters. You get a `Local<String>` `property` and an appropriately typed `args` object that can act like the `args` argument to a `NAN_METHOD` call.

You can use `NanReturnNull()`, `NanReturnEmptyString()`, `NanReturnUndefined()` and `NanReturnValue()` in a `NAN_GETTER`.

<a name="api_nan_setter"></a>
### NAN_SETTER(methodname)

Use `NAN_SETTER` to declare your V8 accessible setters. Same as `NAN_GETTER` but you also get a `Local<Value>` `value` object to work with.

<a name="api_nan_property_getter"></a>
### NAN_PROPERTY_GETTER(cbname)
Use `NAN_PROPERTY_GETTER` to declare your V8 accessible property getters. You get a `Local<String>` `property` and an appropriately typed `args` object that can act similar to the `args` argument to a `NAN_METHOD` call.

You can use `NanReturnNull()`, `NanReturnEmptyString()`, `NanReturnUndefined()` and `NanReturnValue()` in a `NAN_PROPERTY_GETTER`.

<a name="api_nan_property_setter"></a>
### NAN_PROPERTY_SETTER(cbname)
Use `NAN_PROPERTY_SETTER` to declare your V8 accessible property setters. Same as `NAN_PROPERTY_GETTER` but you also get a `Local<Value>` `value` object to work with.

<a name="api_nan_property_enumerator"></a>
### NAN_PROPERTY_ENUMERATOR(cbname)
Use `NAN_PROPERTY_ENUMERATOR` to declare your V8 accessible property enumerators. You get an appropriately typed `args` object like the `args` argument to a `NAN_PROPERTY_GETTER` call.

You can use `NanReturnNull()`, `NanReturnEmptyString()`, `NanReturnUndefined()` and `NanReturnValue()` in a `NAN_PROPERTY_ENUMERATOR`.

<a name="api_nan_property_deleter"></a>
### NAN_PROPERTY_DELETER(cbname)
Use `NAN_PROPERTY_DELETER` to declare your V8 accessible property deleters. Same as `NAN_PROPERTY_GETTER`.

You can use `NanReturnNull()`, `NanReturnEmptyString()`, `NanReturnUndefined()` and `NanReturnValue()` in a `NAN_PROPERTY_DELETER`.

<a name="api_nan_property_query"></a>
### NAN_PROPERTY_QUERY(cbname)
Use `NAN_PROPERTY_QUERY` to declare your V8 accessible property queries. Same as `NAN_PROPERTY_GETTER`.

You can use `NanReturnNull()`, `NanReturnEmptyString()`, `NanReturnUndefined()` and `NanReturnValue()` in a `NAN_PROPERTY_QUERY`.

<a name="api_nan_index_getter"></a>
### NAN_INDEX_GETTER(cbname)
Use `NAN_INDEX_GETTER` to declare your V8 accessible index getters. You get a `uint32_t` `index` and an appropriately typed `args` object that can act similar to the `args` argument to a `NAN_METHOD` call.

You can use `NanReturnNull()`, `NanReturnEmptyString()`, `NanReturnUndefined()` and `NanReturnValue()` in a `NAN_INDEX_GETTER`.

<a name="api_nan_index_setter"></a>
### NAN_INDEX_SETTER(cbname)
Use `NAN_INDEX_SETTER` to declare your V8 accessible index setters. Same as `NAN_INDEX_GETTER` but you also get a `Local<Value>` `value` object to work with.

<a name="api_nan_index_enumerator"></a>
### NAN_INDEX_ENUMERATOR(cbname)
Use `NAN_INDEX_ENUMERATOR` to declare your V8 accessible index enumerators. You get an appropriately typed `args` object like the `args` argument to a `NAN_INDEX_GETTER` call.

You can use `NanReturnNull()`, `NanReturnEmptyString()`, `NanReturnUndefined()` and `NanReturnValue()` in a `NAN_INDEX_ENUMERATOR`.

<a name="api_nan_index_deleter"></a>
### NAN_INDEX_DELETER(cbname)
Use `NAN_INDEX_DELETER` to declare your V8 accessible index deleters. Same as `NAN_INDEX_GETTER`.

You can use `NanReturnNull()`, `NanReturnEmptyString()`, `NanReturnUndefined()` and `NanReturnValue()` in a `NAN_INDEX_DELETER`.

<a name="api_nan_index_query"></a>
### NAN_INDEX_QUERY(cbname)
Use `NAN_INDEX_QUERY` to declare your V8 accessible index queries. Same as `NAN_INDEX_GETTER`.

You can use `NanReturnNull()`, `NanReturnEmptyString()`, `NanReturnUndefined()` and `NanReturnValue()` in a `NAN_INDEX_QUERY`.

<a name="api_nan_weak_callback"></a>
### NAN_WEAK_CALLBACK(cbname)

Use `NAN_WEAK_CALLBACK` to define your V8 WeakReference callbacks. Do not use for declaration. There is an argument object `const _NanWeakCallbackData<T, P> &data` allowing access to the weak object and the supplied parameter through its `GetValue` and `GetParameter` methods.

```c++
NAN_WEAK_CALLBACK(weakCallback) {
  int *parameter = data.GetParameter();
  NanMakeCallback(NanGetCurrentContext()->Global(), data.GetValue(), 0, NULL);
  if ((*parameter)++ == 0) {
    data.Revive();
  } else {
    delete parameter;
    data.Dispose();
  }
}
```

<a name="api_nan_deprecated"></a>
### NAN_DEPRECATED
Declares a function as deprecated.

```c++
static NAN_DEPRECATED NAN_METHOD(foo) {
  ...
}
```

<a name="api_nan_inline"></a>
### NAN_INLINE
Inlines a function.

```c++
NAN_INLINE int foo(int bar) {
  ...
}
```

<a name="api_nan_new"></a>
### Local&lt;T&gt; NanNew&lt;T&gt;( ... )

Use `NanNew` to construct almost all v8 objects and make new local handles.

```c++
Local<String> s = NanNew<String>("value");

...

Persistent<Object> o;

...

Local<Object> lo = NanNew(o);

```

<a name="api_nan_undefined"></a>
### Handle&lt;Primitive&gt; NanUndefined()

Use instead of `Undefined()`

<a name="api_nan_null"></a>
### Handle&lt;Primitive&gt; NanNull()

Use instead of `Null()`

<a name="api_nan_true"></a>
### Handle&lt;Primitive&gt; NanTrue()

Use instead of `True()`

<a name="api_nan_false"></a>
### Handle&lt;Primitive&gt; NanFalse()

Use instead of `False()`

<a name="api_nan_return_value"></a>
### NanReturnValue(Handle&lt;Value&gt;)

Use `NanReturnValue` when you want to return a value from your V8 accessible method:

```c++
NAN_METHOD(Foo::Bar) {
  ...

  NanReturnValue(NanNew<String>("FooBar!"));
}
```

No `return` statement required.

<a name="api_nan_return_undefined"></a>
### NanReturnUndefined()

Use `NanReturnUndefined` when you don't want to return anything from your V8 accessible method:

```c++
NAN_METHOD(Foo::Baz) {
  ...

  NanReturnUndefined();
}
```

<a name="api_nan_return_null"></a>
### NanReturnNull()

Use `NanReturnNull` when you want to return `Null` from your V8 accessible method:

```c++
NAN_METHOD(Foo::Baz) {
  ...

  NanReturnNull();
}
```

<a name="api_nan_return_empty_string"></a>
### NanReturnEmptyString()

Use `NanReturnEmptyString` when you want to return an empty `String` from your V8 accessible method:

```c++
NAN_METHOD(Foo::Baz) {
  ...

  NanReturnEmptyString();
}
```

<a name="api_nan_scope"></a>
### NanScope()

The introduction of `isolate` references for many V8 calls in Node 0.11 makes `NanScope()` necessary, use it in place of `HandleScope scope`:

```c++
NAN_METHOD(Foo::Bar) {
  NanScope();

  NanReturnValue(NanNew<String>("FooBar!"));
}
```

<a name="api_nan_escapable_scope"></a>
### NanEscapableScope()

The separation of handle scopes into escapable and inescapable scopes makes `NanEscapableScope()` necessary, use it in place of `HandleScope scope` when you later wish to `Close()` the scope:

```c++
Handle<String> Foo::Bar() {
  NanEscapableScope();

  return NanEscapeScope(NanNew<String>("FooBar!"));
}
```

<a name="api_nan_esacpe_scope"></a>
### Local&lt;T&gt; NanEscapeScope(Handle&lt;T&gt; value);
Use together with `NanEscapableScope` to escape the scope. Corresponds to `HandleScope::Close` or `EscapableHandleScope::Escape`.

<a name="api_nan_locker"></a>
### NanLocker()

The introduction of `isolate` references for many V8 calls in Node 0.11 makes `NanLocker()` necessary, use it in place of `Locker locker`:

```c++
NAN_METHOD(Foo::Bar) {
  NanLocker();
  ...
  NanUnlocker();
}
```

<a name="api_nan_unlocker"></a>
### NanUnlocker()

The introduction of `isolate` references for many V8 calls in Node 0.11 makes `NanUnlocker()` necessary, use it in place of `Unlocker unlocker`:

```c++
NAN_METHOD(Foo::Bar) {
  NanLocker();
  ...
  NanUnlocker();
}
```

<a name="api_nan_get_internal_field_pointer"></a>
### void * NanGetInternalFieldPointer(Handle&lt;Object&gt;, int)

Gets a pointer to the internal field with at `index` from a V8 `Object` handle.

```c++
Local<Object> obj;
...
NanGetInternalFieldPointer(obj, 0);
```
<a name="api_nan_set_internal_field_pointer"></a>
### void NanSetInternalFieldPointer(Handle&lt;Object&gt;, int, void *)

Sets the value of the internal field at `index` on a V8 `Object` handle.

```c++
static Persistent<Function> dataWrapperCtor;
...
Local<Object> wrapper = NanPersistentToLocal(dataWrapperCtor)->NewInstance();
NanSetInternalFieldPointer(wrapper, 0, this);
```

<a name="api_nan_object_wrap_handle"></a>
### Local&lt;Object&gt; NanObjectWrapHandle(Object)

When you want to fetch the V8 object handle from a native object you've wrapped with Node's `ObjectWrap`, you should use `NanObjectWrapHandle`:

```c++
NanObjectWrapHandle(iterator)->Get(NanSymbol("end"))
```

<a name="api_nan_symbol"></a>
### String NanSymbol(char *)

Use to create string symbol objects (i.e. `v8::String::NewSymbol(x)`), for getting and setting object properties, or names of objects.

```c++
bool foo = false;
if (obj->Has(NanSymbol("foo")))
  foo = optionsObj->Get(NanSymbol("foo"))->BooleanValue()
```

<a name="api_nan_get_pointer_safe"></a>
### Type NanGetPointerSafe(Type *[, Type])

A helper for getting values from optional pointers. If the pointer is `NULL`, the function returns the optional default value, which defaults to `0`.  Otherwise, the function returns the value the pointer points to.

```c++
char *plugh(uint32_t *optional) {
  char res[] = "xyzzy";
  uint32_t param = NanGetPointerSafe<uint32_t>(optional, 0x1337);
  switch (param) {
    ...
  }
  NanSetPointerSafe<uint32_t>(optional, 0xDEADBEEF);
}  
```

<a name="api_nan_set_pointer_safe"></a>
### bool NanSetPointerSafe(Type *, Type)

A helper for setting optional argument pointers. If the pointer is `NULL`, the function simply returns `false`.  Otherwise, the value is assigned to the variable the pointer points to.

```c++
const char *plugh(size_t *outputsize) {
  char res[] = "xyzzy";
  if !(NanSetPointerSafe<size_t>(outputsize, strlen(res) + 1)) {
    ...
  }

  ...
}
```

<a name="api_nan_raw_string"></a>
### void* NanRawString(Handle&lt;Value&gt;, enum Nan::Encoding, size_t *, void *, size_t, int)

When you want to convert a V8 `String` to a `char*` buffer, use `NanRawString`. You have to supply an encoding as well as a pointer to a variable that will be assigned the number of bytes in the returned string. It is also possible to supply a buffer and its length to the function in order not to have a new buffer allocated. The final argument allows setting `String::WriteOptions`.
Just remember that you'll end up with an object that you'll need to `delete[]` at some point unless you supply your own buffer:

```c++
size_t count;
void* decoded = NanRawString(args[1], Nan::BASE64, &count, NULL, 0, String::HINT_MANY_WRITES_EXPECTED);
char param_copy[count];
memcpy(param_copy, decoded, count);
delete[] decoded;
```

<a name="api_nan_c_string"></a>
### char* NanCString(Handle&lt;Value&gt;, size_t *[, char *, size_t, int])

When you want to convert a V8 `String` to a null-terminated C `char*` use `NanCString`. The resulting `char*` will be UTF-8-encoded, and you need to supply a pointer to a variable that will be assigned the number of bytes in the returned string. It is also possible to supply a buffer and its length to the function in order not to have a new buffer allocated. The final argument allows optionally setting `String::WriteOptions`, which default to `v8::String::NO_OPTIONS`.
Just remember that you'll end up with an object that you'll need to `delete[]` at some point unless you supply your own buffer:

```c++
size_t count;
char* name = NanCString(args[0], &count);
```

<a name="api_nan_boolean_option_value"></a>
### bool NanBooleanOptionValue(Handle&lt;Value&gt;, Handle&lt;String&gt;[, bool])

When you have an "options" object that you need to fetch properties from, boolean options can be fetched with this pair. They check first if the object exists (`IsEmpty`), then if the object has the given property (`Has`) then they get and convert/coerce the property to a `bool`.

The optional last parameter is the *default* value, which is `false` if left off:

```c++
// `foo` is false unless the user supplies a truthy value for it
bool foo = NanBooleanOptionValue(optionsObj, NanSymbol("foo"));
// `bar` is true unless the user supplies a falsy value for it
bool bar = NanBooleanOptionValueDefTrue(optionsObj, NanSymbol("bar"), true);
```

<a name="api_nan_uint32_option_value"></a>
### uint32_t NanUInt32OptionValue(Handle&lt;Value&gt;, Handle&lt;String&gt;, uint32_t)

Similar to `NanBooleanOptionValue`, use `NanUInt32OptionValue` to fetch an integer option from your options object. Can be any kind of JavaScript `Number` and it will be coerced to an unsigned 32-bit integer.

Requires all 3 arguments as a default is not optional:

```c++
uint32_t count = NanUInt32OptionValue(optionsObj, NanSymbol("count"), 1024);
```

<a name="api_nan_error"></a>
### NanError(message), NanTypeError(message), NanRangeError(message)

For making `Error`, `TypeError` and `RangeError` objects.

```c++
Local<Value> res = NanError("you must supply a callback argument");
```

<a name="api_nan_throw_error"></a>
### NanThrowError(message), NanThrowTypeError(message), NanThrowRangeError(message), NanThrowError(Local&lt;Value&gt;), NanThrowError(Local&lt;Value&gt;, int)

For throwing `Error`, `TypeError` and `RangeError` objects. You should `return` this call:

```c++
return NanThrowError("you must supply a callback argument");
```

Can also handle any custom object you may want to throw. If used with the error code argument, it will add the supplied error code to the error object as a property called `code`.

<a name="api_nan_new_buffer_handle"></a>
### Local&lt;Object&gt; NanNewBufferHandle(char *, uint32_t), Local&lt;Object&gt; NanNewBufferHandle(uint32_t)

The `Buffer` API has changed a little in Node 0.11, this helper provides consistent access to `Buffer` creation:

```c++
NanNewBufferHandle((char*)value.data(), value.size());
```

Can also be used to initialize a `Buffer` with just a `size` argument.

Can also be supplied with a `NanFreeCallback` and a hint for the garbage collector.

<a name="api_nan_buffer_use"></a>
### Local&lt;Object&gt; NanBufferUse(char*, uint32_t)

`Buffer::New(char*, uint32_t)` prior to 0.11 would make a copy of the data.
While it was possible to get around this, it required a shim by passing a
callback. So the new API `Buffer::Use(char*, uint32_t)` was introduced to remove
needing to use this shim.

`NanBufferUse` uses the `char*` passed as the backing data, and will free the
memory automatically when the weak callback is called. Keep this in mind, as
careless use can lead to "double free or corruption" and other cryptic failures.

<a name="api_nan_has_instance"></a>
### bool NanHasInstance(Persistent&lt;FunctionTemplate&gt;&, Handle&lt;Value&gt;)

Can be used to check the type of an object to determine it is of a particular class you have already defined and have a `Persistent<FunctionTemplate>` handle for.

<a href="#api_nan_new_context_handle">
### Local&lt;Context&gt; NanNewContextHandle([ExtensionConfiguration*, Handle&lt;ObjectTemplate&gt;, Handle&lt;Value&gt;])
Creates a new `Local<Context>` handle.

```c++
Local<FunctionTemplate> ftmpl = NanNew<FunctionTemplate>();
Local<ObjectTemplate> otmpl = ftmpl->InstanceTemplate();
Local<Context> ctx =  NanNewContextHandle(NULL, otmpl);
```

<a href="#api_nan_get_current_context">
### Local<Context> NanGetCurrentContext()

Gets the current context.

```c++
Local<Context> ctx = NanGetCurrentContext();
```

<a name="api_nan_dispose_persistent"></a>
### void NanDisposePersistent(Persistent&lt;T&gt; &)

Use `NanDisposePersistent` to dispose a `Persistent` handle.

```c++
NanDisposePersistent(persistentHandle);
```

<a name="api_nan_assign_persistent"></a>
### NanAssignPersistent(type, handle, object)

Use `NanAssignPersistent` to assign a non-`Persistent` handle to a `Persistent` one. You can no longer just declare a `Persistent` handle and assign directly to it later, you have to `Reset` it in Node 0.11, so this makes it easier.

In general it is now better to place anything you want to protect from V8's garbage collector as properties of a generic `Object` and then assign that to a `Persistent`. This works in older versions of Node also if you use `NanAssignPersistent`:

```c++
Persistent<Object> persistentHandle;

...

Local<Object> obj = NanNew<Object>();
obj->Set(NanSymbol("key"), keyHandle); // where keyHandle might be a Local<String>
NanAssignPersistent(Object, persistentHandle, obj)
```

<a name="api_nan_make_weak_persistent"></a>
### NanMakeWeakPersistent(Handle&lt;T&gt; handle, P* parameter, _NanWeakCallbackInfo&lt;T, P&gt;::Callback callback)

Creates a weak persistent handle with the supplied parameter and `NAN_WEAK_CALLBACK`. The callback has to be fully specialized to work on all versions of Node.

```c++
NAN_WEAK_CALLBACK(weakCallback) {

...

}

Local<Function> func;

...

int *parameter = new int(0);
NanMakeWeakPersistent(func, parameter, &weakCallback<Function, int>);
```

<a name="api_nan_set_template"></a>
### NanSetTemplate(templ, name, value)

Use to add properties on object and function templates.

<a name="api_nan_make_callback"></a>
### NanMakeCallback(target, func, argc, argv)

Use instead of `node::MakeCallback` to call javascript functions. This is the only proper way of calling functions.

<a name="api_nan_compile_script"></a>
### NanCompileScript(Handle<String> s [, const ScriptOrigin&amp; origin])

Use to create new scripts bound to the current context.

<a name="api_nan_run_script"></a>
### NanRunScript(script)

Use to run both bound and unbound scripts.

<a name="api_nan_adjust_external_memory"></a>
### NanAdjustExternalMemory(int change_in_bytes)

Simply does `AdjustAmountOfExternalAllocatedMemory`

<a name="api_nan_add_gc_epilogue_callback"></a>
### NanAddGCEpilogueCallback(GCEpilogueCallback callback, GCType gc_type_filter=kGCTypeAll)

Simply does `AddGCEpilogueCallback`

<a name="api_nan_add_gc_prologue_callback"></a>
### NanAddGCPrologueCallback(GCPrologueCallback callback, GCType gc_type_filter=kGCTypeAll)

Simply does `AddGCPrologueCallback`

<a name="api_nan_remove_gc_epilogue_callback"></a>
### NanRemoveGCEpilogueCallback(GCEpilogueCallback callback)

Simply does `RemoveGCEpilogueCallback`

<a name="api_nan_add_gc_prologue_callback"></a>
### NanRemoveGCPrologueCallback(GCPrologueCallback callback)

Simply does `RemoveGCPrologueCallback`

<a name="api_nan_get_heap_statistics"></a>
### NanGetHeapStatistics(HeapStatistics *heap_statistics)

Simply does `GetHeapStatistics`

<a name="api_nan_callback"></a>
### NanCallback

Because of the difficulties imposed by the changes to `Persistent` handles in V8 in Node 0.11, creating `Persistent` versions of your `Handle<Function>` is annoyingly tricky. `NanCallback` makes it easier by taking your handle, making it persistent until the `NanCallback` is deleted and even providing a handy `Call()` method to fetch and execute the callback `Function`.

```c++
Local<Function> callbackHandle = args[0].As<Function>();
NanCallback *callback = new NanCallback(callbackHandle);
// pass `callback` around and it's safe from GC until you:
delete callback;
```

You can execute the callback like so:

```c++
// no arguments:
callback->Call(0, NULL);

// an error argument:
Handle<Value> argv[] = {
  NanError(NanNew<String>("fail!"))
};
callback->Call(1, argv);

// a success argument:
Handle<Value> argv[] = {
  NanNull(),
  NanNew<String>("w00t!")
};
callback->Call(2, argv);
```

`NanCallback` also has a `Local<Function> GetCallback()` method that you can use
to fetch a local handle to the underlying callback function, as well  as a
`void SetFunction(Handle<Function>)` for setting the callback on the
`NanCallback`.  Additionally a generic constructor is available for using
`NanCallback` without performing heap allocations.

<a name="api_nan_async_worker"></a>
### NanAsyncWorker

`NanAsyncWorker` is an abstract class that you can subclass to have much of the annoying async queuing and handling taken care of for you. It can even store arbitrary V8 objects for you and have them persist while the async work is in progress.

See a rough outline of the implementation:

```c++
class NanAsyncWorker {
public:
  NanAsyncWorker (NanCallback *callback);

  // Clean up persistent handles and delete the *callback
  virtual ~NanAsyncWorker ();

  // Check the `char *errmsg` property and call HandleOKCallback()
  // or HandleErrorCallback depending on whether it has been set or not
  virtual void WorkComplete ();

  // You must implement this to do some async work. If there is an
  // error then allocate `errmsg` to a message and the callback will
  // be passed that string in an Error object
  virtual void Execute ();

  // Save a V8 object in a Persistent handle to protect it from GC
  void SavePersistent(const char *key, Local<Object> &obj);

  // Fetch a stored V8 object (don't call from within `Execute()`)
  Local<Object> GetFromPersistent(const char *key);

protected:
  // Set this if there is an error, otherwise it's NULL
  const char *errmsg;

  // Default implementation calls the callback function with no arguments.
  // Override this to return meaningful data
  virtual void HandleOKCallback ();

  // Default implementation calls the callback function with an Error object
  // wrapping the `errmsg` string
  virtual void HandleErrorCallback ();
};
```

<a name="api_nan_async_queue_worker"></a>
### NanAsyncQueueWorker(NanAsyncWorker *)

`NanAsyncQueueWorker` will run a `NanAsyncWorker` asynchronously via libuv. Both the *execute* and *after_work* steps are taken care of for you&mdash;most of the logic for this is embedded in `NanAsyncWorker`.

### Contributors

NAN is only possible due to the excellent work of the following contributors:

<table><tbody>
<tr><th align="left">Rod Vagg</th><td><a href="https://github.com/rvagg">GitHub/rvagg</a></td><td><a href="http://twitter.com/rvagg">Twitter/@rvagg</a></td></tr>
<tr><th align="left">Benjamin Byholm</th><td><a href="https://github.com/kkoopa/">GitHub/kkoopa</a></td></tr>
<tr><th align="left">Trevor Norris</th><td><a href="https://github.com/trevnorris">GitHub/trevnorris</a></td><td><a href="http://twitter.com/trevnorris">Twitter/@trevnorris</a></td></tr>
<tr><th align="left">Nathan Rajlich</th><td><a href="https://github.com/TooTallNate">GitHub/TooTallNate</a></td><td><a href="http://twitter.com/TooTallNate">Twitter/@TooTallNate</a></td></tr>
<tr><th align="left">Brett Lawson</th><td><a href="https://github.com/brett19">GitHub/brett19</a></td><td><a href="http://twitter.com/brett19x">Twitter/@brett19x</a></td></tr>
<tr><th align="left">Ben Noordhuis</th><td><a href="https://github.com/bnoordhuis">GitHub/bnoordhuis</a></td><td><a href="http://twitter.com/bnoordhuis">Twitter/@bnoordhuis</a></td></tr>
</tbody></table>

Licence &amp; copyright
-----------------------

Copyright (c) 2014 NAN contributors (listed above).

Native Abstractions for Node.js is licensed under an MIT +no-false-attribs license. All rights not explicitly granted in the MIT license are reserved. See the included LICENSE file for more details.
