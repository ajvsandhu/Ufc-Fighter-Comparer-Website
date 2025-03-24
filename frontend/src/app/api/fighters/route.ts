import { NextResponse } from 'next/server'
import { getDb } from '@/lib/db'

interface Fighter {
  name: string
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
    const db = getDb()
    
    // Search for fighters where name matches the query
    const fighters = db.prepare(`
      SELECT name, nickname, division
      FROM fighters
      WHERE name LIKE ?
      OR nickname LIKE ?
      LIMIT 5
    `).all(`%${query}%`, `%${query}%`) as Fighter[]

    // Format the results
    const formattedFighters = fighters.map((fighter: Fighter) => {
      const parts = []
      parts.push(fighter.name)
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