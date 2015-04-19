var env = require('../../../spec/environment.js');
var os = require('os');

exports.config = {
  seleniumAddress: env.seleniumAddress,
  specs: ['e2e.js'],
  baseUrl: env.baseUrl,
  plugins: [{
    path: '../index.js',
    outdir: os.tmpdir()
  }]
};
