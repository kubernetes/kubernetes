exports.config = {
  baseUrl: 'http://localhost:8000',
  seleniumAddress: 'http://localhost:4444/wd/hub',
  specs: ['chrome/**/*.js', '../components/**/protractor/*.js']
};
