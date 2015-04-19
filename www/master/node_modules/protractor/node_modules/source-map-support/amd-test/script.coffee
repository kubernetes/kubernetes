define ['browser-source-map-support'], (sourceMapSupport) ->
  sourceMapSupport.install()

  foo = -> throw new Error 'foo'

  try
    foo()
  catch e
    if /\bscript\.coffee\b/.test e.stack
      document.body.appendChild document.createTextNode 'Test passed'
    else
      document.body.appendChild document.createTextNode 'Test failed'
      console.log e.stack
