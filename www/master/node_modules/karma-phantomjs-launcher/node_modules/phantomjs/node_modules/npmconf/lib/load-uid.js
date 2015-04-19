module.exports = loadUid

var getUid = require("uid-number")

// Call in the context of a npmconf object

function loadUid (cb) {
  // if we're not in unsafe-perm mode, then figure out who
  // to run stuff as.  Do this first, to support `npm update npm -g`
  if (!this.get("unsafe-perm")) {
    getUid(this.get("user"), this.get("group"), cb)
  } else {
    process.nextTick(cb)
  }
}
