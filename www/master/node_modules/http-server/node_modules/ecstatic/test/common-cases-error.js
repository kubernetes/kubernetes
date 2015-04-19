module.exports = {
  '404' : {
    code: 404
  },
  'something non-existant' : {
    code: 404
  }
};

if (require.main === module) {
  console.log("ok 1 - test cases (error) included");
}
