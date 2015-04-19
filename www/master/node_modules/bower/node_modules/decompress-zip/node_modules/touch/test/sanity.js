var fs = require("fs")
var touch = require("../touch.js")

function _ (fn) { return function (er) {
  if (er) throw er
  fn()
}}

touch.sync("sync")
touch("async", _(function () {
  console.log("async", fs.statSync("async"))
  console.log("sync", fs.statSync("sync"))

  setTimeout(function () {
    touch.sync("sync")
    touch("async", _(function () {
      console.log("async", fs.statSync("async"))
      console.log("sync", fs.statSync("sync"))
      setTimeout(function () {
        touch.sync("sync")
        touch("async", _(function () {
          console.log("async", fs.statSync("async"))
          console.log("sync", fs.statSync("sync"))
          fs.unlinkSync("sync")
          fs.unlinkSync("async")
        }))
      }, 1000)
    }))
  }, 1000)
}))

