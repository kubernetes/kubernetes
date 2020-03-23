# Release HOWTO

since I forget.


1. Review existing tags and pick new release number

    ```sh
    git tag
    ```

2. Tag locally 

    ```sh
    git tag -a v0.1.0 -m "First release"
    ```

   If things get screwed up, delete the tag with

   ```sh
   git tag -d v0.1.0
   ```

3. Test goreleaser

   TODO: how to install goreleaser

   ```sh
   ./scripts/goreleaser-dryrun.sh
   ```

4. Push

    ```bash
    git push origin v0.1.0
    ```

5. Verify release and edit notes. See https://github.com/client9/misspell/releases
