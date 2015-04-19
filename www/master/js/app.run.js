app.run(['$route', angular.noop])
    .run(function(lodash) {
      // Alias lodash
      window['_'] = lodash;
    });
