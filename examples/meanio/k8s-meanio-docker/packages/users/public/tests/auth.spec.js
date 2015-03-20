'use strict';

(function() {
  // Login Controller Spec
  describe('MEAN controllers', function() {
    describe('LoginCtrl', function() {
      beforeEach(function() {
        jasmine.addMatchers({
          toEqualData: function() {
            return {
              compare: function(actual, expected) {
                return {
                  pass: angular.equals(actual, expected)
                };
              }
            };
          }
        });
      });

      beforeEach(function() {
        module('mean');
        module('mean.system');
        module('mean.users');
      });

      var LoginCtrl,
        scope,
        $rootScope,
        $httpBackend,
        $location;

      beforeEach(inject(function($controller, _$rootScope_, _$location_, _$httpBackend_) {

        scope = _$rootScope_.$new();
        $rootScope = _$rootScope_;

        LoginCtrl = $controller('LoginCtrl', {
          $scope: scope,
          $rootScope: _$rootScope_
        });

        $httpBackend = _$httpBackend_;

        $location = _$location_;

      }));

      afterEach(function() {
        $httpBackend.verifyNoOutstandingExpectation();
        $httpBackend.verifyNoOutstandingRequest();
      });

      it('should login with a correct user and password', function() {

        spyOn($rootScope, '$emit');
        // test expected GET request
        $httpBackend.when('POST', '/login').respond(200, {
          user: 'Fred'
        });
        scope.login();
        $httpBackend.flush();
        // test scope value
        expect($rootScope.user).toEqual('Fred');
        expect($rootScope.$emit).toHaveBeenCalledWith('loggedin');
        expect($location.url()).toEqual('/');
      });



      it('should fail to log in ', function() {
        $httpBackend.expectPOST('/login').respond(400, 'Authentication failed');
        scope.login();
        $httpBackend.flush();
        // test scope value
        expect(scope.loginerror).toEqual('Authentication failed.');

      });
    });

    describe('RegisterCtrl', function() {
      beforeEach(function() {
        jasmine.addMatchers({
          toEqualData: function() {
            return {
              compare: function(actual, expected) {
                return {
                  pass: angular.equals(actual, expected)
                };
              }
            };
          }
        });
      });

      beforeEach(function() {
        module('mean');
        module('mean.system');
        module('mean.users');
      });

      var RegisterCtrl,
        scope,
        $rootScope,
        $httpBackend,
        $location;

      beforeEach(inject(function($controller, _$rootScope_, _$location_, _$httpBackend_) {

        scope = _$rootScope_.$new();
        $rootScope = _$rootScope_;

        RegisterCtrl = $controller('RegisterCtrl', {
          $scope: scope,
          $rootScope: _$rootScope_
        });

        $httpBackend = _$httpBackend_;

        $location = _$location_;

      }));

      afterEach(function() {
        $httpBackend.verifyNoOutstandingExpectation();
        $httpBackend.verifyNoOutstandingRequest();
      });

      it('should register with correct data', function() {

        spyOn($rootScope, '$emit');
        // test expected GET request
        scope.user.name = 'Fred';
        $httpBackend.when('POST', '/register').respond(200, 'Fred');
        scope.register();
        $httpBackend.flush();
        // test scope value
        expect($rootScope.user.name).toBe('Fred');
        expect(scope.registerError).toEqual(0);
        expect($rootScope.$emit).toHaveBeenCalledWith('loggedin');
        expect($location.url()).toBe('/');
      });



      it('should fail to register with duplicate Username', function() {
        $httpBackend.when('POST', '/register').respond(400, 'Username already taken');
        scope.register();
        $httpBackend.flush();
        // test scope value
        expect(scope.usernameError).toBe('Username already taken');
        expect(scope.registerError).toBe(null);
      });

      it('should fail to register with non-matching passwords', function() {
        $httpBackend.when('POST', '/register').respond(400, 'Password mismatch');
        scope.register();
        $httpBackend.flush();
        // test scope value
        expect(scope.usernameError).toBe(null);
        expect(scope.registerError).toBe('Password mismatch');
      });
    });

    describe('ForgotPasswordCtrl', function() {
      beforeEach(function() {
        jasmine.addMatchers({
          toEqualData: function() {
            return {
              compare: function(actual, expected) {
                return {
                  pass: angular.equals(actual, expected)
                };
              }
            };
          }
        });
      });

      beforeEach(function() {
        module('mean');
        module('mean.system');
        module('mean.users');
      });

      var ForgotPasswordCtrl,
          scope,
          $rootScope,
          $httpBackend ;

      beforeEach(inject(function($controller, _$rootScope_, _$httpBackend_) {

        scope = _$rootScope_.$new();
        $rootScope = _$rootScope_;

        ForgotPasswordCtrl = $controller('ForgotPasswordCtrl', {
          $scope: scope,
          $rootScope: _$rootScope_
        });

        $httpBackend = _$httpBackend_;

      }));

      afterEach(function() {
        $httpBackend.verifyNoOutstandingExpectation();
        $httpBackend.verifyNoOutstandingRequest();
      });

      it('should display success response on success', function() {
        scope.user.email = 'test@test.com';
        $httpBackend.when('POST', '/forgot-password').respond(200,'Mail successfully sent');
        scope.forgotpassword();
        $httpBackend.flush();

        expect(scope.response).toEqual('Mail successfully sent');

      });
      it('should display error response on failure', function() {
        scope.user.email = 'test@test.com';
        $httpBackend.when('POST', '/forgot-password').respond(400,'User does not exist');
        scope.forgotpassword();
        $httpBackend.flush();

        expect(scope.response).toEqual('User does not exist');

      });

    });
  });


}());
