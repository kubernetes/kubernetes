if (module !== require.main) {
  throw new Error("This file should not be loaded with require()")
}

if (!process.getuid || !process.getgid) {
  throw new Error("this file should not be called without uid/gid support")
}

var argv = process.argv.slice(2)
  , user = argv[0] || process.getuid()
  , group = argv[1] || process.getgid()

if (!isNaN(user)) user = +user
if (!isNaN(group)) group = +group

console.error([user, group])

try {
  process.setgid(group)
  process.setuid(user)
  console.log(JSON.stringify({uid:+process.getuid(), gid:+process.getgid()}))
} catch (ex) {
  console.log(JSON.stringify({error:ex.message,errno:ex.errno}))
}
