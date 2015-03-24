# inject
--
    import "github.com/codegangsta/inject"

inject包提供了多种对实体的映射和依赖注入方式。

## 用法

#### func  InterfaceOf

```go
func InterfaceOf(value interface{}) reflect.Type
```
函数InterfaceOf返回指向接口类型的指针。如果传入的value值不是指向接口的指针，将抛出一个panic异常。

#### type Applicator

```go
type Applicator interface {
    // 在Type map中维持对结构体中每个域的引用并用'inject'来标记
    // 如果注入失败将会返回一个error.
    Apply(interface{}) error
}
```

Applicator接口表示到结构体的依赖映射关系。

#### type Injector

```go
type Injector interface {
    Applicator
    Invoker
    TypeMapper
    // SetParent用来设置父injector. 如果在当前injector的Type map中找不到依赖，
    // 将会继续从它的父injector中找，直到返回error.
    SetParent(Injector)
}
```

Injector接口表示对结构体、函数参数的映射和依赖注入。

#### func  New

```go
func New() Injector
```
New创建并返回一个Injector.

#### type Invoker

```go
type Invoker interface {
    // Invoke尝试将interface{}作为一个函数来调用，并基于Type为函数提供参数。
    // 它将返回reflect.Value的切片，其中存放原函数的返回值。
    // 如果注入失败则返回error.
    Invoke(interface{}) ([]reflect.Value, error)
}
```

Invoker接口表示通过反射进行函数调用。

#### type TypeMapper

```go
type TypeMapper interface {
    // 基于调用reflect.TypeOf得到的类型映射interface{}的值。
    Map(interface{}) TypeMapper
    // 基于提供的接口的指针映射interface{}的值。
    // 该函数仅用来将一个值映射为接口，因为接口无法不通过指针而直接引用到。
    MapTo(interface{}, interface{}) TypeMapper
    // 为直接插入基于类型和值的map提供一种可能性。
    // 它使得这一类直接映射成为可能：无法通过反射直接实例化的类型参数，如单向管道。
    Set(reflect.Type, reflect.Value) TypeMapper
    // 返回映射到当前类型的Value. 如果Type没被映射，将返回对应的零值。
    Get(reflect.Type) reflect.Value
}
```

TypeMapper接口用来表示基于类型到接口值的映射。


## 译者

张强 (qqbunny@yeah.net)