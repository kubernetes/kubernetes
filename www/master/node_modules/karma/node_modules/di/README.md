# Dependency Injection for Node.js

Heavily influenced by [AngularJS] and its implementation of dependency injection.
Inspired by [Guice] and [Pico Container].

[AngularJS]: http://angularjs.org/
[Pico Container]: http://picocontainer.codehaus.org/
[Guice]: http://code.google.com/p/google-guice/

<!--
Differences compare to Angular:
- service -> type
- no config/runtime phase
- no providers (configuration happens by registering config)
- no $provide
- no global module register
- no array annotations (but annotate helper)
- no decorators
- no child injectors (yet)
- comment annotation (TBD)
- node module injection (TBD)
-->
