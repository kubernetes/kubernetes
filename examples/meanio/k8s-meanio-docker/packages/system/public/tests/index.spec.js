'use strict';

(function() {
  describe('MEAN controllers', function() {
    describe('IndexController', function() {
      beforeEach(function() {
        module('mean');
        module('mean.system');
      });

      var scope, IndexController;

      beforeEach(inject(function($controller, $rootScope) {
        scope = $rootScope.$new();

        IndexController = $controller('IndexController', {
          $scope: scope
        });
      }));

      it('should expose some global scope', function() {
        expect(scope.global).toBeTruthy();
      });
    });
  });
})();
