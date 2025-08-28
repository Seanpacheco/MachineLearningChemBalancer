import React from 'react'
import { BarChart, DonutChart, RadarChart } from '@mantine/charts'
import { Box, Grid, Text, Paper, Group } from '@mantine/core'

interface ChartsDashboardProps {
  formValues: {
    phLevel: number
    alkalinityLevel: number | null
    sanitizerLevel: number
    calciumLevel: number | null
    cyanuricAcidLevel: number | null
    targets: {
      ph: number
      alkalinity: number
      chlorine: number
      calcium_hardness: number
      cyanuric_acid: number
    }
  }
  recommendations: Array<{
    parameter: string
    chemical: string
    dosage: number
    unit: string
  }>
}

export const ChartsDashboard: React.FC<ChartsDashboardProps> = ({
  formValues,
  recommendations,
}) => {
  // Bar Chart: Current vs Target Levels
  const barData = [
    {
      parameter: 'pH',
      current: formValues.phLevel,
      target: formValues.targets.ph,
      status:
        Math.abs(formValues.phLevel - formValues.targets.ph) < 0.2
          ? 'good'
          : 'needs-adjustment',
    },
    {
      parameter: 'Alkalinity',
      current: formValues.alkalinityLevel ?? 80,
      target: formValues.targets.alkalinity,
      status:
        Math.abs(
          (formValues.alkalinityLevel ?? 80) - formValues.targets.alkalinity
        ) < 10
          ? 'good'
          : 'needs-adjustment',
    },
    {
      parameter: 'Chlorine',
      current: formValues.sanitizerLevel,
      target: formValues.targets.chlorine,
      status:
        Math.abs(formValues.sanitizerLevel - formValues.targets.chlorine) < 0.5
          ? 'good'
          : 'needs-adjustment',
    },
    {
      parameter: 'Calcium',
      current: formValues.calciumLevel ?? 220,
      target: formValues.targets.calcium_hardness,
      status:
        Math.abs(
          (formValues.calciumLevel ?? 220) - formValues.targets.calcium_hardness
        ) < 20
          ? 'good'
          : 'needs-adjustment',
    },
    {
      parameter: 'Cyanuric Acid',
      current: formValues.cyanuricAcidLevel ?? 40,
      target: formValues.targets.cyanuric_acid,
      status:
        Math.abs(
          (formValues.cyanuricAcidLevel ?? 40) -
            formValues.targets.cyanuric_acid
        ) < 5
          ? 'good'
          : 'needs-adjustment',
    },
  ]

  // Donut Chart: Chemical Dosage Distribution
  const donutData =
    recommendations.length > 0
      ? recommendations.map((rec, index) => ({
          name: rec.chemical,
          value: rec.dosage,
          color: ['blue.6', 'red.6', 'green.6', 'orange.6', 'purple.6'][
            index % 5
          ],
        }))
      : [{ name: 'No recommendations', value: 100, color: 'gray.4' }]

  // Radar Chart: Balance Overview (percentage of target)
  const getParameterScore = (current: number, target: number, type: string) => {
    const deviation = Math.abs(current - target)
    let maxDeviation

    switch (type) {
      case 'ph':
        maxDeviation = 0.4
        break
      case 'alkalinity':
        maxDeviation = 20
        break
      case 'chlorine':
        maxDeviation = 1.0
        break
      case 'calcium':
        maxDeviation = 50
        break
      case 'cyanuric':
        maxDeviation = 10
        break
      default:
        maxDeviation = 1
    }

    return Math.max(0, Math.min(100, 100 - (deviation / maxDeviation) * 100))
  }

  const radarData = [
    {
      parameter: 'pH',
      current: getParameterScore(
        formValues.phLevel,
        formValues.targets.ph,
        'ph'
      ),
      target: 100,
    },
    {
      parameter: 'Alkalinity',
      current: getParameterScore(
        formValues.alkalinityLevel ?? 80,
        formValues.targets.alkalinity,
        'alkalinity'
      ),
      target: 100,
    },
    {
      parameter: 'Chlorine',
      current: getParameterScore(
        formValues.sanitizerLevel,
        formValues.targets.chlorine,
        'chlorine'
      ),
      target: 100,
    },
    {
      parameter: 'Calcium',
      current: getParameterScore(
        formValues.calciumLevel ?? 220,
        formValues.targets.calcium_hardness,
        'calcium'
      ),
      target: 100,
    },
    {
      parameter: 'Cyanuric Acid',
      current: getParameterScore(
        formValues.cyanuricAcidLevel ?? 40,
        formValues.targets.cyanuric_acid,
        'cyanuric'
      ),
      target: 100,
    },
  ]

  return (
    <Box p="md">
      <Text size="xl" fw={700} mb="lg">
        Pool Chemistry Dashboard
      </Text>

      <Grid>
        {/* Bar Chart - Current vs Target */}
        <Grid.Col span={{ base: 12, md: 6 }}>
          <Paper p="md" shadow="sm" radius="md">
            <Text size="lg" fw={600} mb="sm">
              Current vs Target Levels
            </Text>
            <BarChart
              h={300}
              data={barData}
              dataKey="parameter"
              series={[
                { name: 'current', color: 'blue.6' },
                { name: 'target', color: 'green.6' },
              ]}
              tickLine="xy"
              withLegend
              withTooltip
            />
          </Paper>
        </Grid.Col>

        {/* Donut Chart - Chemical Distribution */}
        <Grid.Col span={{ base: 12, md: 6 }}>
          <Paper p="md" shadow="sm" radius="md">
            <Text size="lg" fw={600} mb="sm">
              Recommended Dosages
            </Text>
            <DonutChart
              h={300}
              data={donutData}
              withLabels
              withTooltip
              chartLabel={recommendations.length > 0 ? 'Chemicals' : 'No Recs'}
            />
          </Paper>
        </Grid.Col>

        {/* Radar Chart - Overall Balance */}
        <Grid.Col span={12}>
          <Paper p="md" shadow="sm" radius="md">
            <Text size="lg" fw={600} mb="sm">
              Pool Chemistry Balance
            </Text>
            <Group justify="space-between" mb="xs">
              <Text size="sm" c="dimmed">
                Percentage of target values
              </Text>
              <Group gap="md">
                <Group gap={4}>
                  <Box
                    w={12}
                    h={12}
                    bg="blue.6"
                    style={{ borderRadius: '50%' }}
                  />
                  <Text size="xs">Current Levels</Text>
                </Group>
                <Group gap={4}>
                  <Box
                    w={12}
                    h={12}
                    bg="green.6"
                    style={{ borderRadius: '50%' }}
                  />
                  <Text size="xs">100% = Target</Text>
                </Group>
              </Group>
            </Group>
            <RadarChart
              h={300}
              data={radarData}
              dataKey="parameter"
              withPolarRadiusAxis
              withPolarGrid
              series={[
                { name: 'current', color: 'blue.4', opacity: 0.3 },
                { name: 'target', color: 'gray.4', opacity: 0.1 },
              ]}
            />
          </Paper>
        </Grid.Col>
      </Grid>
    </Box>
  )
}
