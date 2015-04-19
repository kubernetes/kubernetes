# npmconf

The config thing npm uses

If you are interested in interacting with the config settings that npm
uses, then use this module.

However, if you are writing a new Node.js program, and want
configuration functionality similar to what npm has, but for your
own thing, then I'd recommend using [rc](https://github.com/dominictarr/rc),
which is probably what you want.

If I were to do it all over again, that's what I'd do for npm.  But,
alas, there are many systems depending on many of the particulars of
npm's configuration setup, so it's not worth the cost of changing.

## USAGE

```javascript
var npmconf = require('npmconf')

// pass in the cli options that you read from the cli
// or whatever top-level configs you want npm to use for now.
npmconf.load({some:'configs'}, function (er, conf) {
  // do stuff with conf
  conf.get('some', 'cli') // 'configs'
  conf.get('username') // 'joebobwhatevers'
  conf.set('foo', 'bar', 'user')
  conf.save('user', function (er) {
    // foo = bar is now saved to ~/.npmrc or wherever
  })
})
```
