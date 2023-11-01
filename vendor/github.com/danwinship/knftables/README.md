# knftables: a golang nftables library

This is a library for using nftables from Go.

It is not intended to support arbitrary use cases, but instead
specifically focuses on supporing Kubernetes components which are
using nftables in the way that nftables is supposed to be used (as
opposed to using nftables in a naively-translated-from-iptables way,
or using nftables to do totally valid things that aren't the sorts of
things Kubernetes components are likely to need to do).

It is still under development and is not API stable.

## Usage

Create an `Interface` object to manage operations on a single nftables
table:

```golang
nft, err := knftables.New(knftables.IPv4Family, "my-table")
if err != nil {
        return fmt.Errorf("no nftables support: %v", err)
}
```

You can use the `List`, `ListRules`, and `ListElements` methods on the
`Interface` to check if objects exist. `List` returns the names of
`"chains"`, `"sets"`, or `"maps"` in the table, while `ListElements`
returns `Element` objects and `ListRules` returns *partial* `Rule`
objects.

```golang
chains, err := nft.List(ctx, "chains")
if err != nil {
        return fmt.Errorf("could not list chains: %v", err)
}

FIXME

elements, err := nft.ListElements(ctx, "map", "mymap")
if err != nil {
        return fmt.Errorf("could not list map elements: %v", err)
}

FIXME
```

To make changes, create a `Transaction`, add the appropriate
operations to the transaction, and then call `nft.Run` on it:

```golang
tx := nft.NewTransaction()

tx.Add(&knftables.Chain{
        Name:    "mychain",
        Comment: knftables.PtrTo("this is my chain"),
})
tx.Flush(&knftables.Chain{
        Name: "mychain",
})

var destIP net.IP
var destPort uint16
...
tx.Add(&knftables.Rule{
        Chain: "mychain",
        Rule: knftables.Concat(
                "ip daddr", destIP,
                "ip protocol", "tcp",
                "th port", destPort,
                "jump", destChain,
        )
})

err := nft.Run(context, tx)
```

If any operation in the transaction would fail, then `Run()` will
return an error and the entire transaction will be ignored. You can
use the `knftables.IsNotFound()` and `knftables.IsAlreadyExists()`
methods to check for those well-known error types. In a large
transaction, there is no supported way to determine exactly which
operation failed.

## `knftables.Transaction` operations

`knftables.Transaction` operations correspond to the top-level commands
in the `nft` binary. Currently-supported operations are:

- `tx.Add()`: adds an object, which may already exist, as with `nft add`
- `tx.Create()`: creates an object, which must not already exist, as with `nft create`
- `tx.Flush()`: flushes the contents of a table/chain/set/map, as with `nft flush`
- `tx.Delete()`: deletes an object, as with `nft delete`
- `tx.Insert()`: inserts a rule before another rule, as with `nft insert rule`
- `tx.Replace()`: replaces a rule, as with `nft replace rule`

## Objects

The `Transaction` methods take arguments of type `knftables.Object`.
The currently-supported objects are:

- `Table`
- `Chain`
- `Rule`
- `Set`
- `Map`
- `Element`

Optional fields in objects can be filled in with the help of the
`PtrTo()` function, which just returns a pointer to its argument.

`Concat()` can be used to concatenate a series of strings, `[]string`
arrays, and other arguments (including numbers, `net.IP`s /
`net.IPNet`s, and anything else that can be formatted usefully via
`fmt.Sprintf("%s")`) together into a single string. This is often
useful when constructing `Rule`s.

## `knftables.Fake`

There is a fake (in-memory) implementation of `knftables.Interface`
for use in unit tests. Use `knftables.NewFake()` instead of
`knftables.New()` to create it, and then it should work mostly the
same. See `fake.go` for more details of the public APIs for examining
the current state of the fake nftables database.

Note that at the present time, `fake.Run()` is not actually
transactional, so unit tests that rely on things not being changed if
a transaction fails partway through will not work as expected.

## Missing APIs

Various top-level object types are not yet supported (notably the
"stateful objects" like `counter`).

Most IPTables libraries have an API for "add this rule only if it
doesn't already exist", but that does not seem as useful in nftables
(or at least "in nftables as used by Kubernetes-ish components that
aren't just blindly copying over old iptables APIs"), because chains
tend to have static rules and dynamic sets/maps, rather than having
dynamic rules. If you aren't sure if a chain has the correct rules,
you can just `Flush` it and recreate all of the rules.

I've considered changing the semantics of `tx.Add(obj)` so that
`obj.Handle` is filled in with the new object's handle on return from
`Run()`, for ease of deleting later. (This would be implemented by
using the `--handle` (`-a`) and `--echo` (`-e`) flags to `nft add`.)
However, this would require potentially difficult parsing of the `nft`
output. `ListRules` fills in the handles of the rules it returns, so
it's possible to find out a rule's handle after the fact that way. For
other supported object types, either handles don't exist (`Element`)
or you don't really need to know their handles because it's possible
to delete by name instead (`Table`, `Chain`, `Set`, `Map`).

The "destroy" (delete-without-ENOENT) command that exists in newer
versions of `nft` is not currently supported because it would be
unexpectedly heavyweight to emulate on systems that don't have it, so
it is better (for now) to force callers to implement it by hand.

`ListRules` returns `Rule` objects without the `Rule` field filled in,
because it uses the JSON API to list the rules, but there is no easy
way to convert the JSON rule representation back into plaintext form.
This means that it is only useful when either (a) you know the order
of the rules in the chain, but want to know their handles, or (b) you
can recognize the rules you are looking for by their comments, rather
than the rule bodies.

# Design Notes

The library works by invoking the `nft` binary. "Write" operations are
implemented with the ordinary plain-text API, while "read" operations
are implemented with the JSON API, for parseability.

The fact that the API uses functions and objects (e.g.
`tx.Add(&knftables.Chain{...})`) rather than just specifying everything
as textual input to `nft` (e.g. `tx.Exec("add chain ...")`) is mostly
just because it's _much_ easier to have a fake implementation for unit
tests this way.
