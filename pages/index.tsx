import Layout from 'components/Layout';
import { GetStaticPropsResult } from 'next';
import Link from 'next/link';
import React from 'react';

import { getAllPostMetadata, getPostSlugs } from '../lib/api';
import { Metadata } from '../lib/models';

const IndexPage: React.FC<{ metadataList: Metadata[] }> = ({
  metadataList,
}) => {
  return (
    <Layout title='CS5340 notes' metadataList={metadataList}>
      <p>Hello</p>
    </Layout>
  );
};

export default IndexPage;

export async function getStaticProps(): Promise<
  GetStaticPropsResult<{ metadataList: Metadata[] }>
> {
  const metadataList = await getAllPostMetadata();
  return {
    props: {
      metadataList,
    },
  };
}
