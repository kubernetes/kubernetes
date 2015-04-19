module.exports = {
  init: require('./lib/init'),
  add: require('./lib/add'),
  commit: require('./lib/commit'),
  addRemote: require('./lib/addRemote'),
  push: require('./lib/push'),
  pull: require('./lib/pull'),
  tag: require('./lib/tag'),
  branch: require('./lib/branch'),
  merge: require('./lib/merge'),
  checkout: require('./lib/checkout'),
  rm: require('./lib/rm'),
  reset: require('./lib/reset')
};
