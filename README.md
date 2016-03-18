## word2vec-go
Still working on this...


Go port of word2vec algorithms. Using both the original C source [https://github.com/jbowles/word2vec](https://github.com/jbowles/word2vec) and this go implementation of word2vec [https://github.com/koji-ohki-1974/word2vec](https://github.com/koji-ohki-1974/word2vec) to produce a more idiomatic go project.

Working on a server that can train and query the model.

## Test
Tests are written as new parts of word 2 vec are ported. To run tests with vocab operations (which will take some time to finish, so I skip them by default):

I'm still working on vocab operations tests

```sh
go test -vocabrun=1
```
