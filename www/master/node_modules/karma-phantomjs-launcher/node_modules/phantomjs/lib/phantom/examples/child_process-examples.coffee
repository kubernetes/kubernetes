{spawn, execFile} = require "child_process"

child = spawn "ls", ["-lF", "/rooot"]

child.stdout.on "data", (data) ->
  console.log "spawnSTDOUT:", JSON.stringify data

child.stderr.on "data", (data) ->
  console.log "spawnSTDERR:", JSON.stringify data

child.on "exit", (code) ->
  console.log "spawnEXIT:", code

#child.kill "SIGKILL"

execFile "ls", ["-lF", "/usr"], null, (err, stdout, stderr) ->
  console.log "execFileSTDOUT:", JSON.stringify stdout
  console.log "execFileSTDERR:", JSON.stringify stderr

setTimeout (-> phantom.exit 0), 2000
