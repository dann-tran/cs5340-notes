import fs from 'fs';
import matter from 'gray-matter';
import { join } from 'path';

import { Metadata, Post } from './models';

const postsDirectory = join(process.cwd(), '_posts');

export function getPostSlugs(): string[] {
  return fs
    .readdirSync(postsDirectory)
    .map((filename) => filename.replace(/\.md$/, ''));
}

export function getPostBySlug(slug: string): Post {
  const fullPath = join(postsDirectory, `${slug}.md`);
  const fileContents = fs.readFileSync(fullPath, 'utf8');
  const { data, content } = matter(fileContents);
  return {
    metadata: {
      slug,
      title: data.title,
    },
    markdownBody: content,
  };
}

export function getAllPosts(): Post[] {
  return getPostSlugs().map(getPostBySlug);
}

export function getAllPostTitles(): Metadata[] {
  return getAllPosts().map((post) => post.metadata);
}
