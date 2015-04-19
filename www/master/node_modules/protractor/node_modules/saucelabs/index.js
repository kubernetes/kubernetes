module.exports = process.env.SAUCELABS_COV ?
  require('./lib-cov/SauceLabs') :
  require('./lib/SauceLabs');
