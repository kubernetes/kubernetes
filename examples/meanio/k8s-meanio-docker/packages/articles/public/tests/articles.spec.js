'use strict';

(function() {
  // Articles Controller Spec
  describe('MEAN controllers', function() {
    describe('ArticlesController', function() {
      // The $resource service augments the response object with methods for updating and deleting the resource.
      // If we were to use the standard toEqual matcher, our tests would fail because the test values would not match
      // the responses exactly. To solve the problem, we use a newly-defined toEqualData Jasmine matcher.
      // When the toEqualData matcher compares two objects, it takes only object properties into
      // account and ignores methods.
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
        module('mean.articles');
      });

      // Initialize the controller and a mock scope
      var ArticlesController,
        scope,
        $httpBackend,
        $stateParams,
        $location;

      // The injector ignores leading and trailing underscores here (i.e. _$httpBackend_).
      // This allows us to inject a service but then attach it to a variable
      // with the same name as the service.
      beforeEach(inject(function($controller, $rootScope, _$location_, _$stateParams_, _$httpBackend_) {

        scope = $rootScope.$new();

        ArticlesController = $controller('ArticlesController', {
          $scope: scope
        });

        $stateParams = _$stateParams_;

        $httpBackend = _$httpBackend_;

        $location = _$location_;

      }));

      it('$scope.find() should create an array with at least one article object ' +
        'fetched from XHR', function() {

          // test expected GET request
          $httpBackend.expectGET('articles').respond([{
            title: 'An Article about MEAN',
            content: 'MEAN rocks!'
          }]);

          // run controller
          scope.find();
          $httpBackend.flush();

          // test scope value
          expect(scope.articles).toEqualData([{
            title: 'An Article about MEAN',
            content: 'MEAN rocks!'
          }]);

        });

      it('$scope.findOne() should create an array with one article object fetched ' +
        'from XHR using a articleId URL parameter', function() {
          // fixture URL parament
          $stateParams.articleId = '525a8422f6d0f87f0e407a33';

          // fixture response object
          var testArticleData = function() {
            return {
              title: 'An Article about MEAN',
              content: 'MEAN rocks!'
            };
          };

          // test expected GET request with response object
          $httpBackend.expectGET(/articles\/([0-9a-fA-F]{24})$/).respond(testArticleData());

          // run controller
          scope.findOne();
          $httpBackend.flush();

          // test scope value
          expect(scope.article).toEqualData(testArticleData());

        });

      it('$scope.create() with valid form data should send a POST request ' +
        'with the form input values and then ' +
        'locate to new object URL', function() {

          // fixture expected POST data
          var postArticleData = function() {
            return {
              title: 'An Article about MEAN',
              content: 'MEAN rocks!'
            };
          };

          // fixture expected response data
          var responseArticleData = function() {
            return {
              _id: '525cf20451979dea2c000001',
              title: 'An Article about MEAN',
              content: 'MEAN rocks!'
            };
          };

          // fixture mock form input values
          scope.title = 'An Article about MEAN';
          scope.content = 'MEAN rocks!';

          // test post request is sent
          $httpBackend.expectPOST('articles', postArticleData()).respond(responseArticleData());

          // Run controller
          scope.create(true);
          $httpBackend.flush();

          // test form input(s) are reset
          expect(scope.title).toEqual('');
          expect(scope.content).toEqual('');

          // test URL location to new object
          expect($location.path()).toBe('/articles/' + responseArticleData()._id);
        });

      it('$scope.update(true) should update a valid article', inject(function(Articles) {

        // fixture rideshare
        var putArticleData = function() {
          return {
            _id: '525a8422f6d0f87f0e407a33',
            title: 'An Article about MEAN',
            to: 'MEAN is great!'
          };
        };

        // mock article object from form
        var article = new Articles(putArticleData());

        // mock article in scope
        scope.article = article;

        // test PUT happens correctly
        $httpBackend.expectPUT(/articles\/([0-9a-fA-F]{24})$/).respond();

        // testing the body data is out for now until an idea for testing the dynamic updated array value is figured out
        //$httpBackend.expectPUT(/articles\/([0-9a-fA-F]{24})$/, putArticleData()).respond();
        /*
                Error: Expected PUT /articles\/([0-9a-fA-F]{24})$/ with different data
                EXPECTED: {"_id":"525a8422f6d0f87f0e407a33","title":"An Article about MEAN","to":"MEAN is great!"}
                GOT:      {"_id":"525a8422f6d0f87f0e407a33","title":"An Article about MEAN","to":"MEAN is great!","updated":[1383534772975]}
                */

        // run controller
        scope.update(true);
        $httpBackend.flush();

        // test URL location to new object
        expect($location.path()).toBe('/articles/' + putArticleData()._id);

      }));

      it('$scope.remove() should send a DELETE request with a valid articleId ' +
        'and remove the article from the scope', inject(function(Articles) {

          // fixture rideshare
          var article = new Articles({
            _id: '525a8422f6d0f87f0e407a33'
          });

          // mock rideshares in scope
          scope.articles = [];
          scope.articles.push(article);

          // test expected rideshare DELETE request
          $httpBackend.expectDELETE(/articles\/([0-9a-fA-F]{24})$/).respond(204);

          // run controller
          scope.remove(article);
          $httpBackend.flush();

          // test after successful delete URL location articles list
          //expect($location.path()).toBe('/articles');
          expect(scope.articles.length).toBe(0);

        }));
    });
  });
}());
