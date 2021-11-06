import Layout from 'components/Layout';
import { getAllPostMetadata, getPostBySlug, getPostSlugs } from 'lib/api';
import { Metadata, Post } from 'lib/models';
import { Params } from 'next/dist/server/router';
import React from 'react';
import ReactMarkdown from 'react-markdown';
import rehypeKatex from 'rehype-katex';
import remarkMath from 'remark-math';
import styles from 'styles/Home.module.css';

const PostPage: React.FC<{ metadataList: Metadata[]; post: Post }> = ({
  metadataList,
  post,
}) => (
  <Layout title={post.metadata.title} metadataList={metadataList}>
    <h1>{post.metadata.title}</h1>
    <ReactMarkdown rehypePlugins={[rehypeKatex]} remarkPlugins={[remarkMath]}>
      {post.markdownBody}
    </ReactMarkdown>
  </Layout>
);

export default PostPage;

export async function getStaticProps({ params }: Params) {
  const metadataList = await getAllPostMetadata();
  const post = await getPostBySlug(params.slug);
  return {
    props: { metadataList, post },
  };
}

export async function getStaticPaths() {
  const slugs = getPostSlugs();
  return {
    paths: slugs.map((slug: string) => ({
      params: {
        slug,
      },
    })),
    fallback: false,
  };
}
