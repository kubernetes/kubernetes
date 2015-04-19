var system = require('system'),
    env = system.env,
    key;

for (key in env) {
    if (env.hasOwnProperty(key)) {
        console.log(key + '=' + env[key]);
    }
}
phantom.exit();
