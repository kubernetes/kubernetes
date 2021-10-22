/*
Package tasks enables management and retrieval of tasks from the OpenStack
Imageservice.

Example to List Tasks

  listOpts := tasks.ListOpts{
    Owner: "424e7cf0243c468ca61732ba45973b3e",
  }

  allPages, err := tasks.List(imagesClient, listOpts).AllPages()
  if err != nil {
    panic(err)
  }

  allTasks, err := tasks.ExtractTasks(allPages)
  if err != nil {
    panic(err)
  }

  for _, task := range allTasks {
    fmt.Printf("%+v\n", task)
  }

Example to Get a Task

  task, err := tasks.Get(imagesClient, "1252f636-1246-4319-bfba-c47cde0efbe0").Extract()
  if err != nil {
    panic(err)
  }

  fmt.Printf("%+v\n", task)

Example to Create a Task

  createOpts := tasks.CreateOpts{
    Type: "import",
    Input: map[string]interface{}{
      "image_properties": map[string]interface{}{
        "container_format": "bare",
        "disk_format":      "raw",
      },
      "import_from_format": "raw",
      "import_from":        "https://cloud-images.ubuntu.com/bionic/current/bionic-server-cloudimg-amd64.img",
    },
  }

  task, err := tasks.Create(imagesClient, createOpts).Extract()
  if err != nil {
    panic(err)
  }

  fmt.Printf("%+v\n", task)
*/
package tasks
