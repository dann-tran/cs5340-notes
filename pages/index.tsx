import Layout from 'components/Layout';
import { GetStaticPropsResult } from 'next';
import Link from 'next/link';
import React from 'react';

import { getAllPostMetadata } from '../lib/api';
import { Metadata } from '../lib/models';
import styles from 'styles/Home.module.css';

const IndexPage: React.FC<{ metadataList: Metadata[] }> = ({
  metadataList,
}) => {
  return (
    <Layout title='CS5340 notes' metadataList={metadataList}>
      <h1>CS5340 notes</h1>
      <div>Dan N. Tran</div>
      <section>
        <p>
          This site is my notes for the NUS course CS5340. The notes are
          compiled from various textbooks and lecture/course notes on the topic
          of probabilistic graphical modelling (PGM). Relevant source materials
          are credited at the end of each page in this site.
        </p>
        <p>
          The content and organisation of the topics closely follow the course
          as taught by Prof. Lee Gim Hee in AY2021-22 Semester 1.
        </p>
        <p>
          For any feedback or query, contact me at{' '}
          <a href='mailto: dann.tran@u.nus.edu'>dann.tran@u.nus.edu</a>.
        </p>
      </section>
      <section className={styles['contents']}>
        <h2>Table of Contents</h2>
        <ol>
          {metadataList.map((metadata) => (
            <li key={metadata.slug}>
              <Link href={metadata.slug}>
                <a>{metadata.title}</a>
              </Link>
              <div className={styles['tags']}>{metadata.tags}</div>
            </li>
          ))}
        </ol>
      </section>
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
