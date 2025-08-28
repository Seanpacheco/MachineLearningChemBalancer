import { MantineProvider, createTheme } from '@mantine/core'
import './App.css'
import '@mantine/core/styles.css'
import '@mantine/charts/styles.css'
import { Main } from './features/layout/Main'

const theme = createTheme({
  primaryColor: 'seaGreen',
  colors: {
    seaGreen: [
      '#f0f9f8',
      '#e3f0ee',
      '#c2e1dc',
      '#9dd0c8',
      '#7fc3b8',
      '#6cbbad',
      '#60b8a8',
      '#50a192',
      '#449082',
      '#317d70',
    ],
  },
})

function App() {
  return (
    <MantineProvider theme={theme}>
      <Main />
    </MantineProvider>
  )
}

export default App
