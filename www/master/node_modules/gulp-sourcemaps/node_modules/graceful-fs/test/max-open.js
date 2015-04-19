var test = require('tap').test
var fs = require('../')

test('open lots of stuff', function (t) {
  // Get around EBADF from libuv by making sure that stderr is opened
  // Otherwise Darwin will refuse to give us a FD for stderr!
  process.stderr.write('')

  // How many parallel open()'s to do
  var n = 1024
  var opens = 0
  var fds = []
  var going = true
  var closing = false
  var doneCalled = 0

  for (var i = 0; i < n; i++) {
    go()
  }

  function go() {
    opens++
    fs.open(__filename, 'r', function (er, fd) {
      if (er) throw er
      fds.push(fd)
      if (going) go()
    })
  }

  // should hit ulimit pretty fast
  setTimeout(function () {
    going = false
    t.equal(opens - fds.length, n)
    done()
  }, 100)


  function done () {
    if (closing) return
    doneCalled++

    if (fds.length === 0) {
      //console.error('done called %d times', doneCalled)
      // First because of the timeout
      // Then to close the fd's opened afterwards
      // Then this time, to complete.
      // Might take multiple passes, depending on CPU speed
      // and ulimit, but at least 3 in every case.
      t.ok(doneCalled >= 3)
      return t.end()
    }

    closing = true
    setTimeout(function () {
      // console.error('do closing again')
      closing = false
      done()
    }, 100)

    // console.error('closing time')
    var closes = fds.slice(0)
    fds.length = 0
    closes.forEach(function (fd) {
      fs.close(fd, function (er) {
        if (er) throw er
      })
    })
  }
})
