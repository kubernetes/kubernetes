Use this module to convert a username/groupname to a uid/gid number.

Usage:

```
npm install uid-number
```

Then, in your node program:

```javascript
var uidNumber = require("uid-number")
uidNumber("isaacs", function (er, uid, gid) {
  // gid is null because we didn't ask for a group name
  // uid === 24561 because that's my number.
})
```
