exports.config = {
  seleniumAddress: 'http://localhost:4444/wd/hub',
  specs: ['chrome/**/*.js', '../components/**/protractor/*.js']
};
