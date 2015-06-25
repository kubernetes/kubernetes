app.provider('k8sApi',
             function() {

               var urlBase = '';
               var _namespace = 'default';

               this.setUrlBase = function(value) { urlBase = value; };

               this.setNamespace = function(value) { _namespace = value; };
               this.getNamespace = function() { return _namespace; };

               var _get = function($http, baseUrl, query) {
                 var _fullUrl = baseUrl;

                 if (query !== undefined) {
                   _fullUrl += '/' + query;
                 }

                 return $http.get(_fullUrl);
               };

               this.$get = function($http, $q) {
                 var api = {};

                 api.getUrlBase = function() { return urlBase + '/namespaces/' + _namespace; };

                 api.getPods = function(query) { return _get($http, api.getUrlBase() + '/pods', query); };

                 api.getNodes = function(query) { return _get($http, urlBase + '/nodes', query); };

                 api.getMinions = api.getNodes;

                 api.getServices = function(query) { return _get($http, api.getUrlBase() + '/services', query); };

                 api.getReplicationControllers = function(query) {
                   return _get($http, api.getUrlBase() + '/replicationcontrollers', query)
                 };

                 api.getEvents = function(query) { return _get($http, api.getUrlBase() + '/events', query); };

                 return api;
               };
             })
    .config(function(k8sApiProvider, ENV) {
      if (ENV && ENV['/'] && ENV['/']['k8sApiServer']) {
        k8sApiProvider.setUrlBase(ENV['/']['k8sApiServer']);
      }
    });
