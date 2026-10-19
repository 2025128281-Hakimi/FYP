# How to commit and push (FYP)

Run in the VS Code terminal (Ctrl + `):

```
git add -A
git commit -m "Describe what you changed"
git push
```

Or in the Source Control panel: type a message -> Commit -> Sync Changes.

## Notes
- The GitHub remote is named `FYP` (not `origin`).
- If `git push` says "no upstream", run once: `git push -u FYP main`
- If push is "rejected" (GitHub has changes you don't have), run:
  `git pull FYP main` then `git push` again.
- Datasets (PodcastFiller/, TedLium V1/, *.wav, *.csv, ...) are in .gitignore and are never uploaded.
- Check what will be committed first: `git status`
