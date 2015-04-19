var env = require('../../../spec/environment.js');

exports.config = {
  seleniumAddress: env.seleniumAddress,
  framework: 'jasmine2',
  specs: ['fail_spec.js'],
  baseUrl: env.baseUrl,
  plugins: [{
    tenonIO: {
      options: {
        key: 'YOUR_API_KEY', // ADD YOUR API KEY HERE
        level: 'AA' // WCAG AA OR AAA
      },
      printAll: false
    },
    chromeA11YDevTools: {
      treatWarningsAsFailures: true
    },
    path: '../index.js'
  }]
};
