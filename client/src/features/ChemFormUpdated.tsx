import * as React from 'react'
import {
  Button,
  Group,
  Box,
  Text,
  NumberInput,
  SegmentedControl,
  MultiSelect,
  LoadingOverlay,
  Alert,
  Slider,
  Flex,
  Accordion,
  Space,
  Grid,
  SimpleGrid,
} from '@mantine/core'
import { useForm } from '@mantine/form'
import { zodResolver } from 'mantine-form-zod-resolver'
import { z } from 'zod'
import axios from 'axios'
import classes from './ChemForm.module.css'
import { TypeAnimation } from 'react-type-animation'
import { ChartsDashboard } from './ChartsDashboard'

const chemsToRaise = [
  'Sodium Bicarbonate', // alkalinity raise
  'Sodium Carbonate', // pH raise option 1
  'Sodium Hydroxide', // pH raise option 2
  'Chlorine Gas',
  'Calcium Hypochlorite 67%',
  'Calcium Hypochlorite 75%',
  'Sodium Hypochlorite 12%',
  'Lithium Hypochlorite 35%',
  'Trichlor 90%',
  'Dichlor 56%',
  'Dichlor 62%',
  'Calcium Chloride (77%)',
  'Cyanuric Acid',
]

const chemsToLower = ['Muriatic Acid', 'Sodium Thiosulfate']

const schema = z.object({
  sanitizerType: z.enum(['chlorine', 'bromine']),
  sanitizerLevel: z.number().min(0, 'Sanitizer level must be ≥ 0'),
  phLevel: z.number().min(0, 'ph must be ≥ 0').max(14, 'ph must be ≤ 14'),
  alkalinityLevel: z.number().min(0, 'Alkalinity must be ≥ 0').nullable(),
  calciumLevel: z.number().min(0, 'Calcium hardness must be ≥ 0').nullable(),
  cyanuricAcidLevel: z.number().min(0, 'Cyanuric acid must be ≥ 0').nullable(),
  targets: z.object({
    ph: z.number().min(0).max(14),
    alkalinity: z.number().min(0),
    chlorine: z.number().min(0),
    calcium_hardness: z.number().min(0),
    cyanuric_acid: z.number().min(0),
  }),
  poolVolume: z.number().min(1, 'Must be > 0'),
  availableRaiseChemicals: z.array(z.string()),
  availableLowerChemicals: z.array(z.string()),
})

export const ChemLogFormUpdated = () => {
  const [selectedSanitizer, setSelectedSanitizer] =
    React.useState<string>('chlorine')
  const [loading, setLoading] = React.useState(false)
  const [errorMessage, setErrorMessage] = React.useState('')
  const [successMessage, setSuccessMessage] = React.useState('')
  const [recommendations, setRecommendations] = React.useState<
    Array<{
      parameter: string
      chemical: string
      dosage: number
      unit: string
    }>
  >([])

  const form = useForm({
    initialValues: {
      sanitizerType: 'chlorine',
      sanitizerLevel: 3,
      phLevel: 7.6,
      alkalinityLevel: null,
      calciumLevel: null,
      cyanuricAcidLevel: null,
      targets: {
        ph: 7.4,
        alkalinity: 80,
        chlorine: 3.0,
        calcium_hardness: 220,
        cyanuric_acid: 40,
      },
      poolVolume: 10000,
      availableRaiseChemicals: [...chemsToRaise],
      availableLowerChemicals: [...chemsToLower],
    },
    validate: zodResolver(schema),
  })

  const onSubmit = async (values: typeof form.values) => {
    setLoading(true)
    setErrorMessage('')
    setSuccessMessage('')
    setRecommendations([])

    try {
      const payload = {
        ph: values.phLevel,
        alkalinity: values.alkalinityLevel ?? 80,
        chlorine: values.sanitizerLevel,
        calcium_hardness: values.calciumLevel ?? 220,
        cyanuric_acid: values.cyanuricAcidLevel ?? 40,
        pool_volume: values.poolVolume,
        targets: {
          ph: values.targets.ph,
          alkalinity: values.targets.alkalinity,
          chlorine: values.targets.chlorine,
          calcium_hardness: values.targets.calcium_hardness,
          cyanuric_acid: values.targets.cyanuric_acid,
        },
        available_chemicals: Array.from(
          new Set([
            ...values.availableRaiseChemicals,
            ...values.availableLowerChemicals,
          ])
        ),
      }

      const response = await axios.post(
        import.meta.env.VITE_REACT_APP_API_URL || '/api/predict_dosage',
        payload
      )

      if (response.data && response.data.recommendations) {
        setRecommendations(response.data.recommendations)
        setSuccessMessage('Dosage recommendations have been calculated.')
      } else {
        setErrorMessage('No recommendations returned from server.')
      }
    } catch (err: any) {
      if (err.response && err.response.data && err.response.data.error) {
        setErrorMessage(err.response.data.error)
      } else {
        setErrorMessage(
          'An error occurred while fetching dosage recommendations.'
        )
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="md">
      <Box w="-moz-available" maw={600} mx="auto" mt="md" pos="relative">
        <LoadingOverlay visible={loading} />

        <form onSubmit={form.onSubmit(onSubmit)}>
          <Flex justify="center" gap="lg" direction="column">
            <SegmentedControl
              color="seaGreen"
              value={selectedSanitizer}
              onChange={setSelectedSanitizer}
              data={[
                { label: 'Chlorine', value: 'chlorine' },
                { label: 'Bromine', value: 'bromine' },
              ]}
            />
            <Space h="md" />
            {selectedSanitizer === 'chlorine' ? (
              <Slider
                size="lg"
                labelAlwaysOn
                defaultValue={5}
                max={10}
                step={0.5}
                marks={[
                  { value: 0, label: '0' },
                  { value: 1, label: '1' },
                  { value: 2, label: '2' },
                  { value: 3, label: '3' },
                  { value: 5, label: '5' },
                  { value: 7.5, label: '7.5' },
                  { value: 10, label: '10' },
                ]}
                {...form.getInputProps('sanitizerLevel')}
                key={form.key('sanitizerLevel')}
              />
            ) : (
              <Slider
                size="lg"
                labelAlwaysOn
                defaultValue={10}
                max={20}
                step={0.5}
                marks={[
                  { value: 0, label: '0' },
                  { value: 2, label: '2' },
                  { value: 4, label: '4' },
                  { value: 6, label: '6' },
                  { value: 10, label: '10' },
                  { value: 15, label: '15' },
                  { value: 20, label: '20' },
                ]}
                {...form.getInputProps('sanitizerLevel')}
                key={form.key('sanitizerLevel')}
              />
            )}

            <Flex direction="column">
              <Text size="sm" mt="xl">
                ph
              </Text>
              <Slider
                defaultValue={7.6}
                labelAlwaysOn
                min={7}
                max={8}
                step={0.1}
                marks={[
                  { value: 7, label: '7' },
                  { value: 7.2, label: '7.2' },
                  { value: 7.4, label: '7.4' },
                  { value: 7.6, label: '7.6' },
                  { value: 7.8, label: '7.8' },
                  { value: 8, label: '8' },
                ]}
                {...form.getInputProps('phLevel')}
                key={form.key('phLevel')}
              />
            </Flex>
            <NumberInput
              label="Alkalinity"
              allowNegative={false}
              placeholder="parts per million"
              {...form.getInputProps('alkalinityLevel')}
              key={form.key('alkalinityLevel')}
            />

            <NumberInput
              label="Calcium"
              allowNegative={false}
              placeholder="parts per million"
              {...form.getInputProps('calciumLevel')}
              key={form.key('calciumLevel')}
            />
            <NumberInput
              label="Cynaurc Acid"
              allowNegative={false}
              placeholder="parts per million"
              {...form.getInputProps('cyanuricAcidLevel')}
              key={form.key('cyanuricAcidLevel')}
            />

            <NumberInput
              label="Pool Volume (gallons)"
              placeholder="Enter pool volume"
              {...form.getInputProps('poolVolume')}
              min={1}
              mb="md"
            />

            <Accordion variant="unstyled" classNames={classes}>
              <Accordion.Item key="Target Levels" value="Target Levels">
                <Accordion.Control>Target Levels</Accordion.Control>
                <Accordion.Panel>
                  <NumberInput
                    label="Target ph"
                    {...form.getInputProps('targets.ph')}
                    min={0}
                    max={14}
                    mb="sm"
                  />
                  <NumberInput
                    label="Target Alkalinity (ppm)"
                    {...form.getInputProps('targets.alkalinity')}
                    min={0}
                    mb="sm"
                  />
                  <NumberInput
                    label="Target Chlorine (ppm)"
                    {...form.getInputProps('targets.chlorine')}
                    min={0}
                    mb="sm"
                  />
                  <NumberInput
                    label="Target Calcium Hardness (ppm)"
                    {...form.getInputProps('targets.calcium_hardness')}
                    min={0}
                    mb="sm"
                  />
                  <NumberInput
                    label="Target Cyanuric Acid (ppm)"
                    {...form.getInputProps('targets.cyanuric_acid')}
                    min={0}
                    mb="md"
                  />{' '}
                </Accordion.Panel>
              </Accordion.Item>
            </Accordion>

            <Accordion variant="unstyled" classNames={classes}>
              <Accordion.Item
                key="Available Chemicals"
                value="Available Chemicals"
              >
                <Accordion.Control>Available Chemicals</Accordion.Control>
                <Accordion.Panel>
                  <Text mb="xs">Available Chemicals to Raise Parameters</Text>
                  <MultiSelect
                    data={chemsToRaise}
                    placeholder="Select chemicals to raise levels"
                    {...form.getInputProps('availableRaiseChemicals')}
                    mb="md"
                    searchable
                    clearable
                  />

                  <Text mb="xs">Available Chemicals to Lower Parameters</Text>

                  <MultiSelect
                    data={chemsToLower}
                    placeholder="Select chemicals to lower levels"
                    {...form.getInputProps('availableLowerChemicals')}
                    mb="md"
                    searchable
                    clearable
                  />
                </Accordion.Panel>
              </Accordion.Item>
            </Accordion>
            <Group mt="md">
              <Button type="submit">Get Recommendations</Button>
            </Group>
          </Flex>
        </form>
      </Box>
      <Grid gutter="md">
        <Grid.Col>
          {errorMessage && (
            <Alert color="red" mb="sm" title="Error">
              {errorMessage}
            </Alert>
          )}

          {successMessage && (
            <Alert color="green" mb="sm" title="Success">
              {successMessage}
            </Alert>
          )}

          {recommendations.length > 0 && (
            <Box mt="md" mb="lg">
              <LoadingOverlay visible={loading} />
              <Text mb="sm">Dosage Recommendations:</Text>
              <TypeAnimation
                splitter={(str) => str.split(/(?= )/)}
                sequence={[
                  recommendations
                    .map(
                      (rec) =>
                        `${rec.chemical} for ${rec.parameter.toUpperCase()}: ${rec.dosage} ${rec.unit}\n`
                    )
                    .join('\n'),
                  3000,
                ]}
                speed={{ type: 'keyStrokeDelayInMs', value: 30 }}
                repeat={0}
                style={{
                  whiteSpace: 'pre-line',
                  fontSize: '1.5em',
                  display: 'block',
                  minHeight: '200px',
                }}
              />
              {/* recommendations.map((rec) => (
                <Text key={rec.parameter}>
                  <b>{rec.chemical}</b> for <b>{rec.parameter}</b>: {rec.dosage}{' '}
                  {rec.unit}
                </Text>
              )) */}
              <ChartsDashboard
                formValues={form.values}
                recommendations={recommendations}
              />
            </Box>
          )}
        </Grid.Col>
      </Grid>
    </SimpleGrid>
  )
}
