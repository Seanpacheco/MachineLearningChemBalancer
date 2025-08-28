import { Container, Flex, Affix, ActionIcon, Title } from '@mantine/core'
import { LightDarkToggle } from '../lightDarkToggle/LightDarkToggle'
import { ChemLogFormUpdated } from '../ChemFormUpdated'
import { IconFlask } from '@tabler/icons-react'
import classes from './Main.module.css'

export function Main() {
  return (
    <>
      <header className={classes.header}>
        <Container size="md" className={classes.inner}>
          <Flex justify="center" align="center">
            <ActionIcon size="xl" variant="transparent">
              <IconFlask size={40} />
            </ActionIcon>
            <Title>ChemBalancer</Title>
          </Flex>
        </Container>
      </header>
      <Container size="xl" my="md">
        <ChemLogFormUpdated />
      </Container>
      <Affix position={{ bottom: 20, right: 20 }}>
        <LightDarkToggle />
      </Affix>
    </>
  )
}
