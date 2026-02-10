# knftables: a golang nftables library

This is a library for using nftables from Go.

It is not intended to support arbitrary use cases, but instead
specifically focuses on supporting Kubernetes components which are
using nftables in the way that nftables is supposed to be used (as
opposed to using nftables in a naively-translated-from-iptables way,
or using nftables to do totally valid things that aren't the sorts of
things Kubernetes components are likely to need to do; see the
"[iptables porting](./docs/iptables-porting.md)" doc for more thoughts
on porting old iptables-based components to nftables.)

knftables is still under development and is not yet API stable. (See the
section on "Possible future changes" below.)

The library is implemented as a wrapper around the `nft` CLI, because
the CLI API is the only well-documented interface to nftables.
Although it would be possible to use netlink directly (and some other
golang-based nftables libraries do this), that would result in an API
that is quite different from all documented examples of nftables usage
(e.g. the man pages and the [nftables wiki](http://wiki.nftables.org/))
because there is no easy way to convert the "standard" representation
of nftables rules into the netlink form.

(Actually, it's not quite true that there's no other usable API: the
`nft` CLI is just a thin wrapper around `libnftables`, and it would be
possible for knftables to use cgo to invoke that library instead of
using an external binary. However, this would be harder to build and
ship, so I'm not bothering with that for now. But this could be done
in the future without needing to change knftables's API.)

knftables requires nft version 1.0.1 or later, because earlier
versions would download and process the entire ruleset regardless of
what you were doing, which, besides being pointlessly inefficient,
means that in some cases, other people using new features in _their_
tables could prevent you from modifying _your_ table. (In particular,
a change in how some rules are generated starting in nft 1.0.3
triggers a crash in nft 0.9.9 and earlier, _even if you aren't looking
at the table containing that rule_.)

## Usage

Create an `Interface` object to manage operations on a single nftables
table:

```golang
nft, err := knftables.New(knftables.IPv4Family, "my-table")
if err != nil {
        return fmt.Errorf("no nftables support: %v", err)
}
```

`knftables.New` also takes a comma-separated list of options after the
family and table name; see the documentation for that function for
more information.

(If you want to operate on multiple tables or multiple nftables
families, you have two options: you can either create separate
`Interface` objects for each table, or you can create a single
`Interface` and pass `""` for the family and table. In that case, you
will need to explicitly fill in the `Family` and `Table` fields of
every `Chain`, `Rule`, etc, object you create.)

You can use the various `List*` methods on the `Interface` to check if
objects exist. `ListAll` returns a map of the names of top-level
objects in the table, sorted by object type, while `List` returns just
the names of objects of a single type. `ListElements`, `ListRules`,
and `ListCounters` returned parsed objects of the given types. Note
that `ListRules` returns *partial* `Rule` objects; it does not fill in
the `Rule` field.

```golang
allChains, err := nft.List(ctx, "chains")
if err != nil {
        return fmt.Errorf("could not list chains: %v", err)
}
for chain := range sets.New(allChains...).Difference(expectedChains) {
        tx.Delete(&knftables.Chain{Name: chain})
}

// ...

elements, err := nft.ListElements(ctx, "map", "mymap")
if err != nil {
        return fmt.Errorf("could not list map elements: %v", err)
}

...
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

(You can also pass a transaction to `nft.Check()`, which uses `nft
--check`, but otherwise behaves the same as `nft.Run()`.)

## `knftables.Transaction` operations

`knftables.Transaction` operations correspond to the top-level commands
in the `nft` binary. Currently-supported operations are:

- `tx.Add()`: creates an object if it does not already exist, as with `nft add`
- `tx.Create()`: creates an object, which must not already exist, as with `nft create`
- `tx.Flush()`: flushes the contents of a table/chain/set/map, as with `nft flush`
- `tx.Reset()`: resets a counter, as with `nft reset`
- `tx.Delete()`: deletes an object, which must exist, as with `nft delete`
- `tx.Destroy()`: deletes an object if it exists, as with `nft destroy`

For `Rule` objects the semantics and operations are slightly different:

- `tx.Add()`: appends a rule to a chain or adds it after an existing rule, as with `nft add rule`
- `tx.Insert()`: prepends a rule to a chain or inserts it before another rule, as with `nft insert rule`
- `tx.Replace()`: replaces a rule, as with `nft replace rule`
- `tx.Delete()`/`tx.Destroy()`: deletes the rule with the given `Handle`, as with `nft delete rule`/`nft destroy rule`

### `Destroy` operations

Actually doing `nft destroy` requires a fairly new kernel (6.3 or
later) and `nft` binary (1.0.8 or later). Trying to run a transaction
containing a `Destroy` operation on an older host will result in an
error.

There are two construct-time options to help out with this. First, you
can specify `RequireDestroy`, if you want knftables construction to
fail on older hosts:

```golang
nft, err := knftables.New(knftables.IPv4Family, "my-table", knftables.RequireDestroy)
if err != nil {
        ...
```

Alternatively, you can construct the `Interface` with the
`EmulateDestroy` option:

```golang
nft, err := knftables.New(knftables.IPv4Family, "my-table", knftables.EmulateDestroy)
```

in which case knftables will attempt to emulate `nft destroy` if it is
not available by doing a combination of an `add` and a `delete` (where
the `add` will succeed whether the object previously existed or not,
and then the `delete` will succeed because the object definitely
exists at that point). To ensure that this emulation will work, if
`EmulateDestroy` is in effect then `tx.Destroy()` will require that
you pass it an object that is suitable for passing to both `tx.Add()`
and `tx.Delete()` (even if the system you are currently on supports
`nft destroy`). In particular, this means that when `EmulateDestroy`
is in effect:

  - You can only `Destroy()` objects by `Name` or `Key`, not by
    `Handle`.

  - You can't `Destroy()` a `Rule` (since `Rule`s can only be deleted
    by `Handle`).

  - If you include optional fields in the object (e.g. base chain
    properties), they need to be correct (since an `Add()` would fail
    if you passed different values). However, note that you *can* just
    leave the optional fields unset.

  - When `Destroy()`ing a `Set` or `Map` you must include the correct
    `Type` (since an `Add()` would fail if you did not specify it or
    specified it incorrectly).

  - When `Destroy()`ing a `Map` `Element` you must include the correct
    `Value` (since an `Add()` would fail if you did not specify it or
    specified it incorrectly).

## Objects

The `Transaction` methods take arguments of type `knftables.Object`.
The currently-supported objects are:

- `Table`
- `Flowtable`
- `Chain`
- `Rule`
- `Set`
- `Map`
- `Element`
- `Counter`

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

## Missing APIs

Various top-level object types are not yet supported.

Most IPTables libraries have an API for "add this rule only if it
doesn't already exist", but that does not seem as useful in nftables
(or at least "in nftables as used by Kubernetes-ish components that
aren't just blindly copying over old iptables APIs"), because chains
tend to have static rules and dynamic sets/maps, rather than having
dynamic rules. If you aren't sure if a chain has the correct rules,
you can just `Flush` it and recreate all of the rules.

`ListRules` returns `Rule` objects without the `Rule` field filled in,
because it uses the JSON API to list the rules, but there is no easy
way to convert the JSON rule representation back into plaintext form.
This means that it is only useful when either (a) you know the order
of the rules in the chain, but want to know their handles, or (b) you
can recognize the rules you are looking for by their comments, rather
than the rule bodies.

## Possible future changes

### `nft` output parsing

`nft`'s output is documented and standardized, so it ought to be
possible for us to extract better error messages in the event of a
transaction failure.

Additionally, if we used the `--echo` (`-e`) and `--handle` (`-a`)
flags, we could learn the handles associated with newly-created
objects in a transaction, and return these to the caller somehow.
(E.g., by setting the `Handle` field in the object that had been
passed to `tx.Add` when the transaction is run.)

(For now, `ListRules` fills in the handles of the rules it returns, so
it's possible to find out a rule's handle after the fact that way. For
other supported object types, either handles don't exist (`Element`)
or you don't really need to know their handles because it's possible
to delete by name instead (`Table`, `Chain`, `Set`, `Map`).)

### List APIs

The fact that `List` works completely differently from `ListRules` and
`ListElements` is a historical artifact.

I would like to have a single function

```golang
List[T Object](ctx context.Context, template T) ([]T, error)
```

So you could say

```golang
elements, err := nft.List(ctx, &knftables.Element{Set: "myset"})
```

to list the elements of "myset". But this doesn't actually compile
("`syntax error: method must have no type parameters`") because
allowing that would apparently introduce extremely complicated edge
cases in Go generics.

### Set/map type representation

There is currently an annoying asymmetry in the representation of
concatenated types between `Set`/`Map` and `Element`, where the former
uses a string containing `nft` syntax, and the latter uses an array:

```golang
tx.Add(&knftables.Set{
        Name: "firewall",
        Type: "ipv4_addr . inet_proto . inet_service",
})
tx.Add(&knftables.Element{
        Set: "firewall",
        Key: []string{"10.1.2.3", "tcp", "80"},
})
```

This will probably be fixed at some point, which may result in a
change to how the `type` vs `typeof` distinction is handled as well.

### Optimization and rule representation

We will need to optimize the performance of large transactions. One
change that is likely is to avoid pre-concatenating rule elements in
cases like:

```golang
tx.Add(&knftables.Rule{
        Chain: "mychain",
        Rule: knftables.Concat(
                "ip daddr", destIP,
                "ip protocol", "tcp",
                "th port", destPort,
                "jump", destChain,
        )
})
```

This will presumably require a change to `knftables.Rule` and/or
`knftables.Concat()` but I'm not sure exactly what it will be.

## Community, discussion, contribution, and support

knftables is maintained by [Kubernetes SIG Network](https://github.com/kubernetes/community/tree/master/sig-network).

- [sig-network slack channel](https://kubernetes.slack.com/messages/sig-network)
- [kubernetes-sig-network mailing list](https://groups.google.com/forum/#!forum/kubernetes-sig-network)

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for more information about
contributing. Participation in the Kubernetes community is governed by
the [Kubernetes Code of Conduct](code-of-conduct.md).
