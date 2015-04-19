//
// Stub out `require` in rhino
//
function require(arg) {
    var split = arg.split('/');
    var resultModule = split.length == 1 ? less.modules[split[0]] : less[split[1]];
    if (!resultModule) {
        throw { message: "Cannot find module '" + arg + "'"};
    }
    return resultModule;
}

