import { NextResponse } from 'next/server'
import { supabase } from '@/lib/db'

interface Fighter {
  fighter_name: string
  nickname: string | null
  division: string | null
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url)
  const query = searchParams.get('query')

  if (!query) {
    return NextResponse.json({ fighters: [] })
  }

  try {
    const { data: fighters, error } = await supabase
      .from('fighters')
      .select('fighter_name, nickname, division')
      .or(`fighter_name.ilike.%${query}%,nickname.ilike.%${query}%`)
      .limit(5)

    if (error) throw error

    // Format the results
    const formattedFighters = (fighters as Fighter[]).map((fighter: Fighter) => {
      const parts = []
      parts.push(fighter.fighter_name)
      if (fighter.nickname) parts.push(`"${fighter.nickname}"`)
      if (fighter.division) parts.push(fighter.division)
      return parts.join(' ')
    })

    return NextResponse.json({ fighters: formattedFighters })
  } catch (error) {
    console.error('Database error:', error)
    return NextResponse.json(
      { error: 'Failed to search fighters' },
      { status: 500 }
    )
  }
} 