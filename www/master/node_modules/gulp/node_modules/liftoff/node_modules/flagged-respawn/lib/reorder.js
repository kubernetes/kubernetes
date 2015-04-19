module.exports = function (flags, argv) {
  if (!argv) {
    argv = process.argv;
  }
  var args = [argv[1]];
  argv.slice(2).forEach(function (arg) {
    var flag = arg.split('=')[0];
    if (flags.indexOf(flag) !== -1) {
      args.unshift(flag);
    } else {
      args.push(arg);
    }
  });
  args.unshift(argv[0]);
  return args;
};
