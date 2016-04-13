# Docker principles

In the design and development of Docker we try to follow these principles:

(Work in progress)

* Don't try to replace every tool. Instead, be an ingredient to improve them.
* Less code is better.
* Fewer components are better. Do you really need to add one more class?
* 50 lines of straightforward, readable code is better than 10 lines of magic that nobody can understand.
* Don't do later what you can do now. "//FIXME: refactor" is not acceptable in new code.
* When hesitating between 2 options, choose the one that is easier to reverse.
* No is temporary, Yes is forever. If you're not sure about a new feature, say no. You can change your mind later.
* Containers must be portable to the greatest possible number of machines. Be suspicious of any change which makes machines less interchangeable.
* The less moving parts in a container, the better.
* Don't merge it unless you document it.
* Don't document it unless you can keep it up-to-date.
* Don't merge it unless you test it!
* Everyone's problem is slightly different. Focus on the part that is the same for everyone, and solve that.
