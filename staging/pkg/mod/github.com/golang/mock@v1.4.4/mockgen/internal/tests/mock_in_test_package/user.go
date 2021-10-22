package users

//go:generate mockgen --source=user.go --destination=mock_test.go --package=users_test

type User struct {
	Name string
}

type Finder interface {
	FindUser(name string) User
	Add(u User)
}
