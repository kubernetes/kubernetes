system = require("system")
env = system.env
key = undefined
for key of env
  console.log key + "=" + env[key]  if env.hasOwnProperty(key)
phantom.exit()